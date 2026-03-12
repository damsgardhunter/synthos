import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "wouter";
import { queryClient } from "@/lib/queryClient";
import { useWebSocket } from "@/hooks/use-websocket";
import type { LearningPhase, ResearchLog, NovelInsight, ConvergenceSnapshot, Milestone } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CheckCircle2, Clock, Loader2, Zap, BookOpen, ArrowRight, BarChart3, FileText, Lightbulb, Sparkles, TrendingUp, TrendingDown, Minus, Target, Gauge, Star, FlaskConical, Trophy, GraduationCap, Layers, Database, BrainCircuit, Map as MapIcon, Notebook, ExternalLink } from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, BarChart, Bar, Cell,
} from "recharts";

function StatusIcon({ status }: { status: string }) {
  if (status === "completed") return <CheckCircle2 className="h-5 w-5 text-green-500" />;
  if (status === "active") return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
  return <Clock className="h-5 w-5 text-muted-foreground" />;
}

const OPEN_ENDED_LABELS: Record<number, string> = {
  3: "bonding patterns analyzed",
  4: "materials indexed",
  5: "property models trained",
  6: "novel candidates generated",
  7: "SC candidates evaluated",
  8: "synthesis processes mapped",
  9: "reactions catalogued",
  10: "physics computations run",
  11: "structures predicted",
  12: "candidates screened",
};

function ActivityBar({ itemsLearned, isActive }: { itemsLearned: number; isActive: boolean }) {
  const milestones = [10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000];
  const safeItems = Number.isFinite(itemsLearned) ? itemsLearned : 0;
  const currentMilestone = milestones.find(m => safeItems < m) ?? milestones[milestones.length - 1];
  const prevMilestone = milestones[milestones.indexOf(currentMilestone) - 1] ?? 0;
  const range = currentMilestone - prevMilestone;
  const progressInRange = range > 0 ? Math.min(100, ((safeItems - prevMilestone) / range) * 100) : 0;

  return (
    <div className="space-y-1">
      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isActive
              ? "bg-blue-500 dark:bg-blue-400"
              : itemsLearned > 0
                ? "bg-primary/60"
                : "bg-muted-foreground/20"
          }`}
          style={{ width: `${safeItems === 0 ? 0 : Math.max(5, progressInRange)}%` }}
        />
      </div>
      {isActive && safeItems > 0 && (
        <p className="text-[10px] text-muted-foreground/70 font-mono">
          next milestone: {currentMilestone.toLocaleString()}
        </p>
      )}
    </div>
  );
}

function PhaseCard({ phase, index }: { phase: LearningPhase; index: number }) {
  const insights = phase.insights ?? [];
  const isCompleted = phase.status === "completed";
  const isActive = phase.status === "active";
  const isFinitePhase = phase.id <= 2;

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
        {isFinitePhase ? (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">Progress</span>
              <span className={`text-xs font-mono font-bold ${isCompleted ? "text-green-600 dark:text-green-400" : "text-primary"}`}>
                {Number.isFinite(phase.progress) ? phase.progress.toFixed(1) : "0.0"}%
              </span>
            </div>
            <Progress
              value={phase.progress}
              className={`h-2 ${isCompleted ? "[&>div]:bg-green-500" : ""}`}
            />
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-mono text-muted-foreground">
                {phase.itemsLearned.toLocaleString()} / {phase.totalItems.toLocaleString()} learned
              </span>
              {isCompleted && (
                <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">
                  COMPLETE
                </Badge>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">Activity</span>
              <span className={`text-xs font-mono font-bold ${
                isActive ? "text-blue-600 dark:text-blue-400" :
                phase.itemsLearned > 0 ? "text-foreground" :
                "text-muted-foreground"
              }`} data-testid={`phase-count-${phase.id}`}>
                {phase.itemsLearned.toLocaleString()} {OPEN_ENDED_LABELS[phase.id] ?? "items processed"}
              </span>
            </div>
            <ActivityBar itemsLearned={phase.itemsLearned} isActive={isActive} />
          </div>
        )}

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

function buildProgressFromPhases(phases: LearningPhase[]) {
  const phaseMap = new Map(phases.map(p => [p.id, p.progress ?? 0]));
  return [
    {
      label: "Current",
      atomic: phaseMap.get(1) ?? 0,
      elements: phaseMap.get(2) ?? 0,
      bonding: phaseMap.get(3) ?? 0,
      materials: phaseMap.get(4) ?? 0,
      prediction: phaseMap.get(5) ?? 0,
      discovery: phaseMap.get(6) ?? 0,
      scResearch: phaseMap.get(7) ?? 0,
      synthesis: phaseMap.get(8) ?? 0,
      reactions: phaseMap.get(9) ?? 0,
      compPhysics: phaseMap.get(10) ?? 0,
      crystalStructures: phaseMap.get(11) ?? 0,
      pipeline: phaseMap.get(12) ?? 0,
    },
  ];
}

const CATEGORY_COLORS: Record<string, string> = {
  "novel-correlation": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
  "new-mechanism": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  "cross-domain": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  "computational-discovery": "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
  "design-principle": "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  "textbook": "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
  "known-pattern": "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
  "incremental": "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
};

function NoveltyBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? "bg-green-500" : score >= 0.4 ? "bg-blue-500" : "bg-gray-400";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-mono">{pct}%</span>
    </div>
  );
}

function computeMomentum(snapshots: ConvergenceSnapshot[]): { icon: typeof TrendingUp; label: string; color: string } {
  if (snapshots.length < 4) return { icon: Minus, label: "Insufficient data", color: "text-muted-foreground" };

  const recent = snapshots.slice(-3);
  const previous = snapshots.slice(-6, -3);
  if (previous.length < 2) return { icon: Minus, label: "Gathering data", color: "text-muted-foreground" };

  const recentAvg = recent.reduce((s, c) => s + (c.bestScore ?? 0), 0) / recent.length;
  const prevAvg = previous.reduce((s, c) => s + (c.bestScore ?? 0), 0) / previous.length;
  const delta = recentAvg - prevAvg;

  if (delta > 0.02) return { icon: TrendingUp, label: "Improving", color: "text-green-600 dark:text-green-400" };
  if (delta < -0.02) return { icon: TrendingDown, label: "Declining", color: "text-red-600 dark:text-red-400" };
  return { icon: Minus, label: "Plateaued", color: "text-yellow-600 dark:text-yellow-400" };
}

function ConvergenceTracker() {
  const { data: snapshots, isLoading } = useQuery<ConvergenceSnapshot[]>({
    queryKey: ["/api/convergence"],
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Convergence Tracker
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-48" />
        </CardContent>
      </Card>
    );
  }

  if (!snapshots || snapshots.length === 0) {
    return (
      <Card data-testid="convergence-tracker">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Convergence Tracker
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground italic" data-testid="convergence-placeholder">
            Convergence data will appear once the engine completes learning cycles
          </p>
        </CardContent>
      </Card>
    );
  }

  const sorted = [...snapshots].sort((a, b) => a.cycle - b.cycle);
  const latest = sorted[sorted.length - 1];
  const momentum = computeMomentum(sorted);
  const MomentumIcon = momentum.icon;

  const chartData = sorted.map(s => ({
    cycle: `C${s.cycle}`,
    bestTc: s.bestTc ?? 0,
    bestPhysicsTc: s.bestPhysicsTc ?? null,
    bestScore: s.bestScore ?? 0,
    avgTopScore: s.avgTopScore ?? 0,
  }));

  const velocityWindow = Math.min(5, sorted.length);
  let tcVelocity = 0;
  let scoreVelocity = 0;
  let diversityVelocity = 0;
  let cyclesSinceImprovement = 0;
  if (sorted.length >= 2) {
    const recent = sorted.slice(-velocityWindow);
    if (recent.length >= 2) {
      const first = recent[0];
      const last = recent[recent.length - 1];
      const cycles = last.cycle - first.cycle;
      if (cycles > 0) {
        tcVelocity = ((last.bestTc ?? 0) - (first.bestTc ?? 0)) / cycles;
        scoreVelocity = ((last.bestScore ?? 0) - (first.bestScore ?? 0)) / cycles;
        diversityVelocity = ((last.familyDiversity ?? 0) - (first.familyDiversity ?? 0)) / cycles;
      }
    }
    let maxTcSoFar = 0;
    let prevScore = 0;
    let lastImprovementCycle = sorted[0].cycle;
    for (let i = 0; i < sorted.length; i++) {
      const s = sorted[i];
      const tc = s.bestTc ?? 0;
      const score = s.bestScore ?? 0;
      if (tc > maxTcSoFar + 1 || score > prevScore + 0.005) {
        lastImprovementCycle = s.cycle;
        maxTcSoFar = Math.max(maxTcSoFar, tc);
      }
      prevScore = score;
    }
    cyclesSinceImprovement = (latest.cycle ?? 0) - lastImprovementCycle;
  }

  function velocityColor(v: number, threshold: number): string {
    if (v > threshold) return "text-green-600 dark:text-green-400";
    if (v < -threshold) return "text-red-600 dark:text-red-400";
    return "text-yellow-600 dark:text-yellow-400";
  }

  return (
    <Card data-testid="convergence-tracker">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Convergence Tracker
          </CardTitle>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-1.5 text-xs font-medium ${momentum.color}`} data-testid="momentum-indicator">
              <MomentumIcon className="h-4 w-4" />
              {momentum.label}
            </div>
            {latest.pipelinePassRate != null && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground" data-testid="pipeline-pass-rate">
                <Gauge className="h-3.5 w-3.5" />
                Pipeline: {(latest.pipelinePassRate * 100).toFixed(1)}% pass rate
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          {latest.topFormula && (
            <div className="rounded-lg border border-border bg-muted/30 p-3" data-testid="best-candidate-callout">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">Best Candidate</p>
              <p className="text-sm font-bold font-mono">{latest.topFormula}</p>
              <div className="flex items-center gap-3 mt-1">
                {latest.bestTc != null && (
                  <span className="text-xs text-muted-foreground">Tc: <span className="font-mono font-bold text-foreground">{latest.bestTc.toFixed(1)}K</span></span>
                )}
                {latest.bestScore != null && (
                  <span className="text-xs text-muted-foreground">Score: <span className="font-mono font-bold text-foreground">{latest.bestScore.toFixed(3)}</span></span>
                )}
              </div>
            </div>
          )}
          <div className="rounded-lg border border-border bg-muted/30 p-3" data-testid="convergence-stats-total">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">Total Candidates</p>
            <p className="text-lg font-bold font-mono">{(latest.candidatesTotal ?? 0).toLocaleString()}</p>
          </div>
          <div className="rounded-lg border border-border bg-muted/30 p-3" data-testid="convergence-stats-cycles">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">Cycles Tracked</p>
            <p className="text-lg font-bold font-mono">{sorted.length}</p>
            {latest.strategyFocus && (
              <p className="text-[10px] text-muted-foreground mt-0.5 truncate">Focus: {latest.strategyFocus}</p>
            )}
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {latest.familyDiversity != null && (
            <div className="rounded-lg border border-border bg-muted/30 p-3" data-testid="convergence-family-diversity">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">Family Diversity</p>
              <p className="text-lg font-bold font-mono">{latest.familyDiversity}<span className="text-xs text-muted-foreground font-normal">/10 families</span></p>
              <p className="text-[10px] text-muted-foreground mt-0.5">Distinct material families in top 50 candidates</p>
            </div>
          )}
          {latest.duplicatesSkipped != null && latest.duplicatesSkipped > 0 && (
            <div className="rounded-lg border border-border bg-muted/30 p-3" data-testid="convergence-duplicates-skipped">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">Duplicates Skipped</p>
              <p className="text-lg font-bold font-mono">{latest.duplicatesSkipped}</p>
              <p className="text-[10px] text-muted-foreground mt-0.5">Redundant formulas filtered this cycle</p>
            </div>
          )}
        </div>

        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="cycle" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis
              yAxisId="tc"
              orientation="left"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              label={{ value: "Tc (K)", angle: -90, position: "insideLeft", style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" } }}
            />
            <YAxis
              yAxisId="score"
              orientation="right"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              label={{ value: "Score", angle: 90, position: "insideRight", style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" } }}
            />
            <Tooltip
              contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px" }}
              formatter={(v: any, name: string) => [
                name === "bestTc" || name === "bestPhysicsTc" ? `${v != null ? Number(v).toFixed(1) : '-'}K` : Number(v).toFixed(3),
                name === "bestTc" ? "Best Tc" : name === "bestPhysicsTc" ? "Physics Tc" : name === "bestScore" ? "Best Score" : "Avg Top-10"
              ]}
            />
            <Line yAxisId="tc" type="monotone" dataKey="bestTc" name="bestTc" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
            <Line yAxisId="tc" type="monotone" dataKey="bestPhysicsTc" name="bestPhysicsTc" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} connectNulls />
            <Line yAxisId="score" type="monotone" dataKey="bestScore" name="bestScore" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
            <Line yAxisId="score" type="monotone" dataKey="avgTopScore" name="avgTopScore" stroke="#6366f1" strokeWidth={1} strokeDasharray="4 4" dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex flex-wrap gap-x-4 gap-y-1 justify-center">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full bg-red-500" />
            <span className="text-xs text-muted-foreground">Best Tc (K)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full bg-green-500" />
            <span className="text-xs text-muted-foreground">Physics Tc (K)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full bg-indigo-500" />
            <span className="text-xs text-muted-foreground">Best Score</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full border border-indigo-500 bg-transparent" />
            <span className="text-xs text-muted-foreground">Avg Top-10</span>
          </div>
        </div>

        {sorted.length >= 2 && (
          <div className="grid gap-3 sm:grid-cols-4 mt-2" data-testid="learning-velocity">
            <div className="rounded-lg border border-border bg-muted/30 p-2.5">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-0.5">Tc Velocity</p>
              <p className={`text-sm font-bold font-mono ${velocityColor(tcVelocity, 0.5)}`} data-testid="velocity-tc">
                {tcVelocity >= 0 ? "+" : ""}{tcVelocity.toFixed(1)}K/cycle
              </p>
            </div>
            <div className="rounded-lg border border-border bg-muted/30 p-2.5">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-0.5">Score Velocity</p>
              <p className={`text-sm font-bold font-mono ${velocityColor(scoreVelocity, 0.003)}`} data-testid="velocity-score">
                {scoreVelocity >= 0 ? "+" : ""}{scoreVelocity.toFixed(4)}/cycle
              </p>
            </div>
            <div className="rounded-lg border border-border bg-muted/30 p-2.5">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-0.5">Diversity Growth</p>
              <p className={`text-sm font-bold font-mono ${velocityColor(diversityVelocity, 0.1)}`} data-testid="velocity-diversity">
                {diversityVelocity >= 0 ? "+" : ""}{diversityVelocity.toFixed(2)}/cycle
              </p>
            </div>
            <div className="rounded-lg border border-border bg-muted/30 p-2.5">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-0.5">Since Improvement</p>
              <p className={`text-sm font-bold font-mono ${cyclesSinceImprovement <= 1 ? "text-green-600 dark:text-green-400" : cyclesSinceImprovement <= 3 ? "text-yellow-600 dark:text-yellow-400" : "text-red-600 dark:text-red-400"}`} data-testid="cycles-since-improvement">
                {cyclesSinceImprovement} cycle{cyclesSinceImprovement !== 1 ? "s" : ""}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

const MILESTONE_ICONS: Record<string, typeof Star> = {
  "new-family": FlaskConical,
  "tc-record": Trophy,
  "pipeline-graduate": GraduationCap,
  "diversity-threshold": Layers,
  "knowledge-milestone": Database,
  "insight-cascade": BrainCircuit,
};

function MilestoneTimeline() {
  const { data, isLoading } = useQuery<{ milestones: Milestone[]; total: number }>({
    queryKey: ["/api/milestones"],
  });

  if (isLoading || !data || data.milestones.length === 0) return null;

  return (
    <Card data-testid="milestone-timeline">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Star className="h-4 w-4 text-amber-500 fill-amber-500" />
            Discovery Milestones
          </CardTitle>
          <Badge variant="secondary" className="text-xs border-0 bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-300" data-testid="milestone-count">
            {data.total} total
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {data.milestones.slice(0, 8).map((ms) => {
            const Icon = MILESTONE_ICONS[ms.type] || Star;
            return (
              <div
                key={ms.id}
                className="flex items-start gap-3 p-2.5 rounded-lg border border-amber-200 dark:border-amber-800/50 bg-amber-50/50 dark:bg-amber-950/20"
                data-testid={`milestone-${ms.id}`}
              >
                <div className="p-1.5 rounded-md bg-amber-100 dark:bg-amber-900/50 flex-shrink-0">
                  <Icon className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-amber-800 dark:text-amber-200">{ms.title}</span>
                    <span className="text-amber-500 flex-shrink-0">
                      {Array.from({ length: ms.significance }, (_, j) => (
                        <Star key={j} className="h-2.5 w-2.5 inline fill-amber-400 text-amber-400" />
                      ))}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{ms.description}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] text-muted-foreground">Cycle {ms.cycle}</span>
                    {ms.relatedFormula && (
                      <Link href={`/candidate/${encodeURIComponent(ms.relatedFormula)}`}>
                        <Badge variant="secondary" className="text-[10px] border-0 h-4 px-1.5 font-mono cursor-pointer hover-elevate flex items-center gap-1" data-testid={`link-candidate-${ms.relatedFormula}`}>
                          {ms.relatedFormula}
                          <ExternalLink className="h-2.5 w-2.5 flex-shrink-0" />
                        </Badge>
                      </Link>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

const FAMILY_COLORS: Record<string, { bg: string; text: string; border: string; tagBg: string }> = {
  "Hydrides":       { bg: "bg-red-500",    text: "text-red-400",    border: "border-red-500/40",    tagBg: "bg-red-500/20" },
  "Cuprates":       { bg: "bg-blue-500",   text: "text-blue-400",   border: "border-blue-500/40",   tagBg: "bg-blue-500/20" },
  "Pnictides":      { bg: "bg-indigo-500", text: "text-indigo-400", border: "border-indigo-500/40", tagBg: "bg-indigo-500/20" },
  "Chalcogenides":  { bg: "bg-amber-500",  text: "text-amber-400",  border: "border-amber-500/40",  tagBg: "bg-amber-500/20" },
  "Borides":        { bg: "bg-emerald-500",text: "text-emerald-400",border: "border-emerald-500/40",tagBg: "bg-emerald-500/20" },
  "Carbides":       { bg: "bg-teal-500",   text: "text-teal-400",   border: "border-teal-500/40",   tagBg: "bg-teal-500/20" },
  "Nitrides":       { bg: "bg-cyan-500",   text: "text-cyan-400",   border: "border-cyan-500/40",   tagBg: "bg-cyan-500/20" },
  "Oxides":         { bg: "bg-orange-500", text: "text-orange-400", border: "border-orange-500/40",  tagBg: "bg-orange-500/20" },
  "Intermetallics": { bg: "bg-purple-500", text: "text-purple-400", border: "border-purple-500/40", tagBg: "bg-purple-500/20" },
  "Sulfides":       { bg: "bg-yellow-500", text: "text-yellow-400", border: "border-yellow-500/40", tagBg: "bg-yellow-500/20" },
  "Other":          { bg: "bg-gray-500",   text: "text-gray-400",   border: "border-gray-500/40",   tagBg: "bg-gray-500/20" },
};

function parseFormulaElements(formula: string): string[] {
  const elements: string[] = [];
  const regex = /([A-Z][a-z]?)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    if (!elements.includes(match[1])) elements.push(match[1]);
  }
  return elements;
}

function heatOpacity(count: number, maxCount: number): number {
  if (maxCount === 0) return 0.15;
  return Math.max(0.15, Math.min(1, count / maxCount));
}

interface CycleCandidate {
  formula: string;
  tc: number;
  passed: boolean;
  reason: string;
  family: string;
}

function KnowledgeMap({ onFamilyClick, selectedFamily }: { onFamilyClick?: (family: string) => void; selectedFamily?: string | null }) {
  const { data: memory, isLoading: memLoading } = useQuery<{
    familyStats: Record<string, { count: number; maxTc: number; avgScore: number }>;
    lastCycleCandidates: CycleCandidate[];
    lastCycleFamilyCounts: Record<string, number>;
    autonomousLoopStats?: {
      inverseOptimizer?: { bestTcAcrossAll: number; activeCampaigns: number };
    };
  }>({
    queryKey: ["/api/engine/memory"],
    refetchInterval: 30000,
  });

  const { messages } = useWebSocket();
  useEffect(() => {
    const engineEvents = messages.filter(m => m.type === "log" && m.data?.phase === "engine");
    if (engineEvents.length > 0) {
      queryClient.invalidateQueries({ queryKey: ["/api/engine/memory"] });
    }
  }, [messages]);

  const familyStats = memory?.familyStats ?? {};
  const cycleCandidates = memory?.lastCycleCandidates ?? [];
  const cycleFamilyCounts = memory?.lastCycleFamilyCounts ?? {};

  const families = Object.entries(familyStats)
    .map(([name, stats]) => ({
      name,
      totalCount: stats.count,
      maxTc: stats.maxTc,
      avgScore: stats.avgScore,
      lastCycleCount: cycleFamilyCounts[name] || 0,
      elements: new Set<string>(),
      colors: FAMILY_COLORS[name] || FAMILY_COLORS["Other"],
    }))
    .sort((a, b) => b.totalCount - a.totalCount);

  for (const c of cycleCandidates) {
    const fam = families.find(f => f.name === c.family);
    if (fam) {
      for (const el of parseFormulaElements(c.formula)) {
        fam.elements.add(el);
      }
    }
  }

  const maxCycleCount = Math.max(1, ...families.map(f => f.lastCycleCount));
  const maxTotalCount = Math.max(1, ...families.map(f => f.totalCount));

  const filteredCycleCandidates = selectedFamily
    ? cycleCandidates.filter(c => c.family === selectedFamily)
    : cycleCandidates;

  return (
    <Card data-testid="knowledge-map">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <MapIcon className="h-4 w-4 text-primary" />
          Knowledge Map
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          Live heatmap of material families under exploration. Intensity reflects last-cycle activity.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {memLoading ? (
          <Skeleton className="h-[260px] w-full" />
        ) : families.length === 0 ? (
          <p className="text-sm text-muted-foreground italic py-8 text-center" data-testid="knowledge-map-empty">
            Knowledge map will populate once candidates are discovered
          </p>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2" data-testid="family-heatmap">
              {families.map(fam => {
                const intensity = heatOpacity(fam.lastCycleCount, maxCycleCount);
                const sizeScale = 0.7 + 0.3 * (fam.totalCount / maxTotalCount);
                const isSelected = selectedFamily === fam.name;
                const isActive = fam.lastCycleCount > 0;

                return (
                  <button
                    key={fam.name}
                    className={`relative rounded-lg border-2 p-2.5 transition-all text-left overflow-hidden ${
                      isSelected ? `${fam.colors.border} ring-2 ring-primary/30` : "border-border/50 hover:border-border"
                    }`}
                    style={{
                      minHeight: `${Math.round(64 * sizeScale)}px`,
                    }}
                    onClick={() => onFamilyClick?.(fam.name)}
                    data-testid={`family-heatmap-${fam.name.toLowerCase()}`}
                  >
                    <div
                      className={`absolute inset-0 ${fam.colors.bg} transition-opacity`}
                      style={{ opacity: intensity * 0.25 }}
                    />
                    <div className="relative z-10">
                      <div className="flex items-center justify-between gap-1">
                        <span className={`text-xs font-semibold ${isActive ? fam.colors.text : "text-muted-foreground"}`}>
                          {fam.name}
                        </span>
                        {isActive && (
                          <span className={`h-2 w-2 rounded-full ${fam.colors.bg} animate-pulse`} />
                        )}
                      </div>
                      <div className="flex items-baseline gap-1.5 mt-1">
                        <span className="text-lg font-bold font-mono" data-testid={`text-family-count-${fam.name.toLowerCase()}`}>
                          {fam.totalCount}
                        </span>
                        <span className="text-[10px] text-muted-foreground">total</span>
                        {fam.lastCycleCount > 0 && (
                          <span className={`text-[10px] font-mono font-semibold ${fam.colors.text}`}>
                            +{fam.lastCycleCount}
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-muted-foreground font-mono mt-0.5">
                        Tc: {Math.round(fam.maxTc)}K
                      </div>
                      {fam.elements.size > 0 && (
                        <div className="flex flex-wrap gap-0.5 mt-1.5">
                          {Array.from(fam.elements).slice(0, 8).map(el => (
                            <span
                              key={el}
                              className={`text-[9px] font-mono px-1 py-px rounded ${fam.colors.tagBg} ${fam.colors.text}`}
                              data-testid={`element-tag-${el}`}
                            >
                              {el}
                            </span>
                          ))}
                          {fam.elements.size > 8 && (
                            <span className="text-[9px] text-muted-foreground">+{fam.elements.size - 8}</span>
                          )}
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>

            {(() => {
              const invOpt = memory?.autonomousLoopStats?.inverseOptimizer;
              const inverseTc = invOpt?.bestTcAcrossAll ?? 0;
              const loopTc = memory?.autonomousLoopStats?.activeLearning?.bestTcFromLoop ?? 0;
              const dbTc = memory?.bestTc ?? 0;
              const unifiedTc = Math.max(dbTc, loopTc, inverseTc);
              if (unifiedTc <= 0) return null;
              return (
                <div className={`flex items-center justify-between p-2.5 rounded-lg border ${unifiedTc > 200 ? "border-amber-500/30 bg-amber-500/5" : "border-border bg-muted/30"}`} data-testid="unified-best-tc">
                  <div className="flex items-center gap-2">
                    <Target className="h-3.5 w-3.5 text-amber-500" />
                    <span className="text-xs font-semibold">Best Tc (All Methods)</span>
                  </div>
                  <span className={`text-sm font-mono font-bold ${unifiedTc > 200 ? "text-amber-600 dark:text-amber-400" : "text-foreground"}`} data-testid="text-unified-best-tc">
                    {Math.round(unifiedTc)}K
                  </span>
                </div>
              );
            })()}

            <div data-testid="last-cycle-candidates">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Last Cycle Candidates ({filteredCycleCandidates.length})
                </h4>
                {selectedFamily && (
                  <Badge variant="outline" className="text-[10px]">{selectedFamily}</Badge>
                )}
              </div>
              {filteredCycleCandidates.length === 0 ? (
                <p className="text-xs text-muted-foreground italic py-4 text-center">
                  {cycleCandidates.length === 0 ? "Waiting for first cycle to complete" : "No candidates in this family last cycle"}
                </p>
              ) : (
                <ScrollArea className="h-[200px]">
                  <div className="space-y-1">
                    {filteredCycleCandidates.map((c, i) => {
                      const colors = FAMILY_COLORS[c.family] || FAMILY_COLORS["Other"];
                      return (
                        <div
                          key={`${c.formula}-${i}`}
                          className={`flex items-center justify-between px-2.5 py-1.5 rounded-md border text-xs transition-colors ${
                            c.passed ? "bg-green-500/5 border-green-500/20" : "bg-muted/30 border-border/50"
                          }`}
                          data-testid={`cycle-candidate-${i}`}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <div className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${c.passed ? "bg-green-500" : "bg-muted-foreground/40"}`} />
                            <Link href={`/candidate/${encodeURIComponent(c.formula)}`}>
                              <span className="font-mono font-semibold text-primary hover:underline cursor-pointer truncate" data-testid={`link-cycle-formula-${i}`}>
                                {c.formula}
                              </span>
                            </Link>
                            <Badge variant="outline" className={`text-[9px] px-1 py-0 ${colors.text} ${colors.border}`}>
                              {c.family}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                            <span className="font-mono text-muted-foreground">
                              {c.tc > 0 ? `${Math.round(c.tc)}K` : "--"}
                            </span>
                            {c.passed ? (
                              <CheckCircle2 className="h-3 w-3 text-green-500" />
                            ) : (
                              <span className="text-[9px] text-muted-foreground/60 max-w-[80px] truncate">{c.reason}</span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </ScrollArea>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

interface CycleNarrative {
  detail: string;
  timestamp: string;
}

function CycleJournal() {
  const { data: memory } = useQuery<{ cycleNarratives: CycleNarrative[] }>({
    queryKey: ["/api/engine/memory"],
    refetchInterval: 30000,
  });

  const narratives = memory?.cycleNarratives ?? [];
  if (narratives.length === 0) return null;

  return (
    <Card data-testid="cycle-journal">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Notebook className="h-4 w-4 text-primary" />
          Cycle Journal
        </CardTitle>
        <p className="text-xs text-muted-foreground">Recent cycle summaries from the learning engine.</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {narratives.slice(0, 5).map((n, i) => (
            <div
              key={i}
              className="px-3 py-2 rounded-md bg-muted/40 border border-border/50"
              data-testid={`narrative-${i}`}
            >
              <p className="text-xs text-foreground leading-relaxed">{n.detail}</p>
              <span className="text-[9px] text-muted-foreground font-mono">
                {new Date(n.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function ResearchPipeline() {
  const { data: phases, isLoading: phasesLoading } = useQuery<LearningPhase[]>({
    queryKey: ["/api/learning-phases"],
  });
  const { data: logs, isLoading: logsLoading } = useQuery<ResearchLog[]>({
    queryKey: ["/api/research-logs"],
  });
  const { data: insightData, isLoading: insightsLoading } = useQuery<{ insights: NovelInsight[]; total: number }>({
    queryKey: ["/api/novel-insights"],
  });
  const [familyFilter, setFamilyFilter] = useState<string | null>(null);

  const ws = useWebSocket();

  useEffect(() => {
    const relevantTypes = ["phaseUpdate", "progress", "insight", "cycleEnd", "log", "convergenceUpdate"];
    const hasRelevant = ws.messages.some((m) => relevantTypes.includes(m.type));
    if (hasRelevant) {
      queryClient.invalidateQueries({ queryKey: ["/api/learning-phases"] });
      queryClient.invalidateQueries({ queryKey: ["/api/research-logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/novel-insights"] });
      queryClient.invalidateQueries({ queryKey: ["/api/superconductor-candidates"] });
      queryClient.invalidateQueries({ queryKey: ["/api/engine/memory"] });
    }
    const hasConvergence = ws.messages.some((m) => m.type === "convergenceUpdate");
    if (hasConvergence) {
      queryClient.invalidateQueries({ queryKey: ["/api/convergence"] });
    }
    const hasMilestone = ws.messages.some((m) => m.type === "milestone");
    if (hasMilestone) {
      queryClient.invalidateQueries({ queryKey: ["/api/milestones"] });
    }
  }, [ws.messages.length]);

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
          {phases && phases.length > 0 ? (() => {
            const progressData = buildProgressFromPhases(phases);
            const barData = [
              { name: "Subatomic", value: progressData[0].atomic, color: "#6366f1" },
              { name: "Elements", value: progressData[0].elements, color: "#8b5cf6" },
              { name: "Bonding", value: progressData[0].bonding, color: "#06b6d4" },
              { name: "Materials", value: progressData[0].materials, color: "#10b981" },
              { name: "Prediction", value: progressData[0].prediction, color: "#f59e0b" },
              { name: "Discovery", value: progressData[0].discovery, color: "#ef4444" },
              { name: "SC Research", value: progressData[0].scResearch, color: "#ec4899" },
              { name: "Synthesis", value: progressData[0].synthesis, color: "#14b8a6" },
              { name: "Reactions", value: progressData[0].reactions, color: "#a855f7" },
              { name: "Comp. Physics", value: progressData[0].compPhysics, color: "#3b82f6" },
              { name: "Crystals", value: progressData[0].crystalStructures, color: "#f97316" },
              { name: "Pipeline", value: progressData[0].pipeline, color: "#84cc16" },
            ];
            return (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={barData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} angle={-35} textAnchor="end" height={60} />
                    <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px" }}
                      formatter={(v: any) => [`${Math.round(v)}%`]}
                    />
                    <Bar dataKey="value" name="Progress" radius={[4, 4, 0, 0]}>
                      {barData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 justify-center">
                  {barData.map(item => (
                    <div key={item.name} className="flex items-center gap-1.5">
                      <div className="h-2 w-4 rounded-full" style={{ background: item.color }} />
                      <span className="text-xs text-muted-foreground">{item.name} {Math.round(item.value)}%</span>
                    </div>
                  ))}
                </div>
              </>
            );
          })() : (
            <p className="text-sm text-muted-foreground italic" data-testid="progress-placeholder">
              Progress data will appear once the engine begins learning
            </p>
          )}
        </CardContent>
      </Card>

      <ConvergenceTracker />

      <div className="grid gap-4 lg:grid-cols-2">
        <KnowledgeMap
          onFamilyClick={(fam) => setFamilyFilter(prev => prev === fam ? null : fam)}
          selectedFamily={familyFilter}
        />
        <CycleJournal />
      </div>

      {familyFilter && (
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs">
            Filtered: {familyFilter}
          </Badge>
          <button
            className="text-xs text-muted-foreground hover:text-foreground underline"
            onClick={() => setFamilyFilter(null)}
            data-testid="clear-family-filter"
          >
            Clear filter
          </button>
        </div>
      )}

      <MilestoneTimeline />

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

      {insightsLoading && (
        <Card>
          <CardContent className="p-6">
            <Skeleton className="h-6 w-48 mb-4" />
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
            </div>
          </CardContent>
        </Card>
      )}

      {insightData && insightData.insights.length > 0 && (
        <Card data-testid="novel-insights-section">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-primary" />
                Insight Novelty Tracker
              </CardTitle>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>{insightData.total} total insights evaluated</span>
                <span className="font-bold text-green-600 dark:text-green-400">
                  {insightData.insights.filter(i => i.isNovel).length} novel
                </span>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Each insight is evaluated for novelty -- textbook knowledge is flagged, while genuinely new correlations and mechanisms are highlighted.
            </p>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px]">
              <div className="space-y-2">
                {insightData.insights.map((insight, i) => (
                  <div
                    key={insight.id ?? i}
                    data-testid={`insight-row-${i}`}
                    className={`p-3 rounded-lg border ${insight.isNovel ? "border-green-200 dark:border-green-900 bg-green-50/50 dark:bg-green-950/20" : "border-border bg-muted/30"}`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 shrink-0">
                        {insight.isNovel ? (
                          <Sparkles className="h-4 w-4 text-green-500" />
                        ) : (
                          <BookOpen className="h-4 w-4 text-muted-foreground" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0 space-y-1.5">
                        <p className={`text-sm leading-relaxed ${insight.isNovel ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                          {insight.insightText}
                        </p>
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge
                            className={`${CATEGORY_COLORS[insight.category ?? "known-pattern"]} border-0 text-[10px]`}
                          >
                            {insight.category ?? "unknown"}
                          </Badge>
                          <span className="text-[10px] text-muted-foreground">
                            {insight.phaseName}
                          </span>
                          {insight.noveltyScore != null && (
                            <NoveltyBar score={insight.noveltyScore} />
                          )}
                          {insight.isNovel && (
                            <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">
                              NOVEL
                            </Badge>
                          )}
                        </div>
                        {insight.noveltyReason && (
                          <p className="text-[10px] text-muted-foreground italic">{insight.noveltyReason}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

