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
  LineChart, Line, BarChart, Bar, Cell, Treemap
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

const FAMILY_BUBBLE_COLORS: Record<string, string> = {
  "Hydrides": "#ef4444",
  "Cuprates": "#3b82f6",
  "Pnictides": "#6366f1",
  "Chalcogenides": "#f59e0b",
  "Borides": "#10b981",
  "Carbides": "#14b8a6",
  "Nitrides": "#06b6d4",
  "Oxides": "#f97316",
  "Intermetallics": "#a855f7",
  "Other": "#9ca3af",
};

function TreemapContent(props: any) {
  const { x, y, width, height, name, count, bestTc, fill } = props;
  if (width < 30 || height < 20) return null;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} rx={4} stroke="hsl(var(--background))" strokeWidth={2} style={{ cursor: "pointer" }} />
      {width > 50 && height > 35 && (
        <>
          <text x={x + width / 2} y={y + height / 2 - 6} textAnchor="middle" fill="white" fontSize={width > 80 ? 11 : 9} fontWeight="bold">{name}</text>
          <text x={x + width / 2} y={y + height / 2 + 8} textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize={8}>{count} cand.</text>
          {bestTc > 0 && height > 50 && (
            <text x={x + width / 2} y={y + height / 2 + 20} textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize={8}>Tc: {Math.round(bestTc)}K</text>
          )}
        </>
      )}
    </g>
  );
}

function KnowledgeMap({ onFamilyClick, selectedFamily }: { onFamilyClick?: (family: string) => void; selectedFamily?: string | null }) {
  const { data: rawData, isLoading } = useQuery<any>({
    queryKey: ["/api/superconductor-candidates"],
  });

  const candidates = Array.isArray(rawData) ? rawData : rawData?.candidates ?? [];

  const familyData = (() => {
    if (!candidates || candidates.length === 0) return [];
    const families: Record<string, { count: number; bestTc: number; bestScore: number }> = {};
    for (const c of candidates) {
      const fam = c.materialFamily ?? "Other";
      if (!families[fam]) families[fam] = { count: 0, bestTc: 0, bestScore: 0 };
      families[fam].count++;
      if ((c.predictedTc ?? 0) > families[fam].bestTc) families[fam].bestTc = c.predictedTc ?? 0;
      if ((c.ensembleScore ?? 0) > families[fam].bestScore) families[fam].bestScore = c.ensembleScore ?? 0;
    }
    return Object.entries(families)
      .map(([name, data]) => ({
        name,
        size: data.count,
        count: data.count,
        bestTc: data.bestTc,
        bestScore: data.bestScore,
        fill: FAMILY_BUBBLE_COLORS[name] ?? "#9ca3af",
      }))
      .sort((a, b) => b.size - a.size);
  })();

  const selectedInfo = selectedFamily ? familyData.find(f => f.name === selectedFamily) : null;

  return (
    <Card data-testid="knowledge-map">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <MapIcon className="h-4 w-4 text-primary" />
          Knowledge Map
        </CardTitle>
        <p className="text-xs text-muted-foreground">Material families explored by the engine. Size reflects candidate count.</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-[260px] w-full" />
        ) : familyData.length === 0 ? (
          <p className="text-sm text-muted-foreground italic py-8 text-center" data-testid="knowledge-map-empty">
            Knowledge map will populate once candidates are discovered
          </p>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={260}>
              <Treemap
                data={familyData}
                dataKey="size"
                aspectRatio={4 / 3}
                content={<TreemapContent />}
                onClick={(node: any) => {
                  if (node?.name && onFamilyClick) onFamilyClick(node.name);
                }}
              />
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-2 mt-3">
              {familyData.map(f => (
                <button
                  key={f.name}
                  className={`flex items-center gap-1.5 text-[10px] px-2 py-0.5 rounded-full border transition-colors ${selectedFamily === f.name ? "border-primary bg-primary/10 font-semibold" : "border-border hover:bg-muted/50"}`}
                  onClick={() => onFamilyClick?.(f.name)}
                  data-testid={`family-filter-${f.name.toLowerCase()}`}
                >
                  <div className="h-2 w-2 rounded-full" style={{ backgroundColor: f.fill }} />
                  {f.name} ({f.count})
                </button>
              ))}
            </div>
            {selectedInfo && (
              <div className="mt-3 p-3 rounded-lg border bg-muted/30" data-testid="family-detail">
                <p className="text-xs font-semibold">{selectedInfo.name}</p>
                <div className="flex gap-4 mt-1 text-[10px] text-muted-foreground">
                  <span>{selectedInfo.count} candidates</span>
                  <span>Best Tc: {Math.round(selectedInfo.bestTc)}K</span>
                  <span>Top score: {selectedInfo.bestScore.toFixed(3)}</span>
                </div>
              </div>
            )}
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

