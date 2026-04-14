import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";
import { queryClient, throttledInvalidate } from "@/lib/queryClient";
import { useWebSocket } from "@/hooks/use-websocket";
import type { LearningPhase, ResearchLog, NovelInsight, ConvergenceSnapshot, Milestone } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
    refetchInterval: false,
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
    avgTop10Tc: (s as any).avgTop10Tc ?? null,
    dftSelectedTc: (s as any).dftSelectedTc ?? null,
    bestScore: s.bestScore ?? 0,
    avgTopScore: s.avgTopScore ?? 0,
    r2Score: s.r2Score ?? null,
  }));

  const velocityWindow = Math.min(5, sorted.length);
  let tcVelocity = 0;
  let scoreVelocity = 0;
  let diversityVelocity = 0;
  let r2Velocity = 0;
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
        if (last.r2Score != null && first.r2Score != null) {
          r2Velocity = (last.r2Score - first.r2Score) / cycles;
        }
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
        {/* Best Candidate callout above graph */}
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
              {latest.r2Score != null && (
                <span className="text-xs text-muted-foreground">R²: <span className="font-mono font-bold text-foreground">{latest.r2Score.toFixed(4)}</span></span>
              )}
            </div>
          </div>
        )}

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
              label={{ value: "Score / R²", angle: 90, position: "insideRight", style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" } }}
            />
            <Tooltip
              contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px" }}
              formatter={(v: any, name: string) => {
                if (name === "bestTc") return [`${v != null ? Number(v).toFixed(1) : '-'}K`, "Best Tc"];
                if (name === "bestPhysicsTc") return [`${v != null ? Number(v).toFixed(1) : '-'}K`, "Physics Tc"];
                if (name === "avgTop10Tc") return [`${v != null ? Number(v).toFixed(1) : '-'}K`, "Avg Top-10 Tc"];
                if (name === "dftSelectedTc") return [`${v != null ? Number(v).toFixed(1) : '-'}K`, "DFT Selected Tc"];
                if (name === "r2Score") return [v != null ? Number(v).toFixed(4) : '-', "GNN R²"];
                const label = name === "bestScore" ? "DFT Score" : "Avg Ensemble";
                return [Number(v).toFixed(3), label];
              }}
            />
            <Line yAxisId="tc" type="monotone" dataKey="bestTc" name="bestTc" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
            <Line yAxisId="tc" type="monotone" dataKey="bestPhysicsTc" name="bestPhysicsTc" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} connectNulls />
            <Line yAxisId="tc" type="monotone" dataKey="avgTop10Tc" name="avgTop10Tc" stroke="#c084fc" strokeWidth={2} dot={{ r: 2 }} connectNulls />
            <Line yAxisId="tc" type="monotone" dataKey="dftSelectedTc" name="dftSelectedTc" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} connectNulls />
            <Line yAxisId="score" type="monotone" dataKey="bestScore" name="bestScore" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
            <Line yAxisId="score" type="monotone" dataKey="avgTopScore" name="avgTopScore" stroke="#6366f1" strokeWidth={1} strokeDasharray="4 4" dot={false} />
            <Line yAxisId="score" type="monotone" dataKey="r2Score" name="r2Score" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} connectNulls />
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
            <div className="h-2 w-4 rounded-full bg-purple-400" />
            <span className="text-xs text-muted-foreground">Avg Top-10 Tc (K)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full bg-cyan-500" />
            <span className="text-xs text-muted-foreground">DFT Selected Tc (K)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full bg-indigo-500" />
            <span className="text-xs text-muted-foreground">DFT Score</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full border border-indigo-500 bg-transparent" />
            <span className="text-xs text-muted-foreground">Avg Ensemble</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-full bg-amber-500" />
            <span className="text-xs text-muted-foreground">GNN R²</span>
          </div>
        </div>

        {/* Total Candidates — centered below legend */}
        <div className="flex justify-center">
          <div className="rounded-lg border border-border bg-muted/30 px-6 py-3 text-center" data-testid="convergence-stats-total">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">Total Candidates</p>
            <p className="text-lg font-bold font-mono">{(latest.candidatesTotal ?? 0).toLocaleString()}</p>
          </div>
        </div>

        {/* Velocity tracking boxes */}
        {sorted.length >= 2 && (
          <div className="grid gap-3 sm:grid-cols-3 mt-2" data-testid="learning-velocity">
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
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-0.5">R² Velocity</p>
              <p className={`text-sm font-bold font-mono ${velocityColor(r2Velocity, 0.005)}`} data-testid="velocity-r2">
                {r2Velocity >= 0 ? "+" : ""}{r2Velocity.toFixed(4)}/cycle
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
    refetchInterval: false,
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
    bestTc?: number;
    autonomousLoopStats?: {
      inverseOptimizer?: { bestTcAcrossAll: number; activeCampaigns: number };
      activeLearning?: { bestTcFromLoop: number };
    };
  }>({
    queryKey: ["/api/engine/memory"],
    refetchInterval: false,
  });

  const { messages, messageCount } = useWebSocket();
  // engine/memory holds cycle-level aggregates — refresh once per cycle, not per candidate log.
  // "log" events with phase="engine" fire ~100×/cycle during screening; that caused 1Hz HTTP spam.

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
    refetchInterval: false,
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
    refetchInterval: false,
  });
  const {
    data: logsData,
    isLoading: logsLoading,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery<{ logs: ResearchLog[]; nextCursor: number | null; hasMore: boolean }>({
    queryKey: ["/api/research-logs"],
    queryFn: async ({ pageParam }) => {
      const url = pageParam
        ? `/api/research-logs?cursor=${pageParam}&limit=50`
        : `/api/research-logs?limit=50`;
      const res = await fetch(url);
      return res.json();
    },
    initialPageParam: null as number | null,
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    refetchInterval: false,
  });
  const logs = useMemo(() => logsData?.pages.flatMap(p => p.logs) ?? [], [logsData]);
  const { data: insightData, isLoading: insightsLoading } = useQuery<{ insights: NovelInsight[]; total: number }>({
    queryKey: ["/api/novel-insights"],
    refetchInterval: false,
  });
  const [familyFilter, setFamilyFilter] = useState<string | null>(null);

  const ws = useWebSocket();

  useEffect(() => {
    const last = ws.messages[ws.messages.length - 1];
    if (!last) return;
    // "progress" fires per batch item (potentially many/sec); "insight" fires per discovery.
    // Only coarse-grained events (phaseUpdate, cycleEnd) should trigger broad invalidation.
    if (last.type === "phaseUpdate") {
      throttledInvalidate("/api/learning-phases");
    }
    if (last.type === "cycleEnd") {
      throttledInvalidate("/api/learning-phases");
      throttledInvalidate("/api/engine/memory");
      throttledInvalidate("/api/research-logs");
      throttledInvalidate("/api/superconductor-candidates");
    }
    if (last.type === "insight" || last.type === "cycleEnd") {
      queryClient.invalidateQueries({ queryKey: ["/api/novel-insights"] });
    }
    if (last.type === "convergenceUpdate") {
      queryClient.invalidateQueries({ queryKey: ["/api/convergence"] });
    }
    if (last.type === "milestone") {
      queryClient.invalidateQueries({ queryKey: ["/api/milestones"] });
    }
  }, [ws.messageCount]);

  const logSentinelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = logSentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) fetchNextPage(); },
      { threshold: 0.1 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  const logsByPhase = useMemo(() => {
    const grouped: Record<string, ResearchLog[]> = {};
    logs?.forEach(log => {
      if (!grouped[log.phase]) grouped[log.phase] = [];
      grouped[log.phase].push(log);
    });
    return grouped;
  }, [logs]);

  const sourceColors: Record<string, string> = {
    "NIST": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
    "Materials Project": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
    "OQMD": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
    "AFLOW": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
    "Internal": "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
    "Internal ML Model": "bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300",
    "DFT Engine": "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
  };

  // Pipeline funnel data (for Multi-Fidelity Pipeline tab)
  const { data: pipelineStatsData } = useQuery<{ pipelineStages: { stage: number; count: number; passed: number }[]; crystalStructures: number; computationalResults: number }>({
    queryKey: ["/api/pipeline-stats"],
  });

  // Synthesis variables data
  const { data: synthesisData } = useQuery<{
    parameterSpace: {
      totalVariables: number;
      categories: { name: string; count: number; parameters: string[] }[];
      totalGridPoints: number;
      discreteVariables: { name: string; options: string[] }[];
    };
    optimizerStats: {
      totalOptimized: number;
      avgFeasibility: number;
      complexityBreakdown: Record<string, number>;
      methodBreakdown: Record<string, number>;
      categoryUsage: Record<string, number>;
      topConditions: { formula: string; method: string; feasibility: number; tc: number }[];
      parameterRangesExplored: Record<string, { min: number; max: number; count: number }>;
    };
  }>({
    queryKey: ["/api/synthesis-variables/stats"],
    refetchInterval: false,
  });

  const STAGE_NAMES = ["ML Filter", "Electronic Structure", "Phonon / E-Ph Coupling", "Tc Prediction (Eliashberg)", "Synthesis Feasibility"];
  const STAGE_COLORS = ["bg-gray-500", "bg-blue-500", "bg-purple-500", "bg-amber-500", "bg-green-500"];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="synthos-heading text-2xl gold-text tracking-wider flex items-center gap-2">
          <FileText className="h-6 w-6 text-[hsl(var(--gold))]" />
          Pipeline Statistics
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Full learning trajectory from subatomic particles to novel material discovery — tracking every phase of scientific understanding.
        </p>
      </div>

      <Tabs defaultValue="learning-phases" className="space-y-4">
        <TabsList className="flex flex-wrap h-auto gap-1">
          <TabsTrigger value="learning-phases">Learning Phases</TabsTrigger>
          <TabsTrigger value="multi-fidelity">Multi-Fidelity Pipeline</TabsTrigger>
          <TabsTrigger value="synthesis-vars">Synthesis Variables</TabsTrigger>
          <TabsTrigger value="advanced-physics">Advanced Physics</TabsTrigger>
        </TabsList>

        <TabsContent value="learning-phases" className="space-y-6">

      <ConvergenceTracker />

      <CycleJournal />

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
                  {logs.map((log, i) => (
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
                  {logs.length === 0 && (
                    <div className="py-8 text-center text-muted-foreground text-sm">No research logs yet</div>
                  )}
                  <div ref={logSentinelRef} className="py-1" />
                  {isFetchingNextPage && (
                    <div className="py-3 flex justify-center">
                      <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    </div>
                  )}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
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

      <div className="space-y-4">
        <h2 className="text-base font-semibold">Learning Phases</h2>
        {phasesLoading ? (
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-48" />)}
          </div>
        ) : (
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
            {phases?.map((phase, i) => <PhaseCard key={phase.id} phase={phase} index={i} />)}
          </div>
        )}
      </div>

        </TabsContent>

        <TabsContent value="multi-fidelity" className="space-y-4">
          <Card className="border-[hsl(var(--gold)/0.2)]">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                Multi-Fidelity Screening Pipeline
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                5-stage computational pipeline: cheap models filter first, expensive methods confirm
              </p>
            </CardHeader>
            <CardContent className="space-y-2">
              {STAGE_NAMES.map((name, i) => {
                const data = pipelineStatsData?.pipelineStages?.find(s => s.stage === i);
                const total = data?.count ?? 0;
                const passed = data?.passed ?? 0;
                const failed = total - passed;
                const maxWidth = Math.max(20, 100 - i * 15);
                return (
                  <div key={i} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <div className={`h-3 w-3 rounded-sm ${STAGE_COLORS[i]}`} />
                        <span className="font-medium">Stage {i}: {name}</span>
                      </div>
                      <div className="flex items-center gap-3 font-mono">
                        <span className="text-green-600 dark:text-green-400">{passed} passed</span>
                        {failed > 0 && <span className="text-red-500">{failed} failed</span>}
                      </div>
                    </div>
                    <div className="relative h-6 bg-muted rounded overflow-hidden" style={{ width: `${maxWidth}%` }}>
                      {total > 0 && (
                        <>
                          <div
                            className={`absolute inset-y-0 left-0 ${STAGE_COLORS[i]} opacity-80 transition-all`}
                            style={{ width: `${(passed / Math.max(total, 1)) * 100}%` }}
                          />
                          <div className="absolute inset-0 flex items-center justify-center text-xs font-mono font-medium text-white mix-blend-difference">
                            {passed}/{total}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <div className="grid grid-cols-3 gap-4">
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4 text-center">
                <p className="text-xs text-muted-foreground">Crystal Structures</p>
                <p className="text-2xl font-mono font-bold">{pipelineStatsData?.crystalStructures ?? 0}</p>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4 text-center">
                <p className="text-xs text-muted-foreground">Computations Run</p>
                <p className="text-2xl font-mono font-bold">{pipelineStatsData?.computationalResults ?? 0}</p>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4 text-center">
                <p className="text-xs text-muted-foreground">Total Stages</p>
                <p className="text-2xl font-mono font-bold">{pipelineStatsData?.pipelineStages?.length ?? 0}</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="synthesis-vars" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Parameter Categories</p>
                <p className="text-2xl font-mono font-bold">{synthesisData?.parameterSpace?.categories?.length ?? 0}</p>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Total Variables</p>
                <p className="text-2xl font-mono font-bold">{synthesisData?.parameterSpace?.totalVariables ?? 0}</p>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Grid Points</p>
                <p className="text-2xl font-mono font-bold">{synthesisData?.parameterSpace?.totalGridPoints ?? 0}</p>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Conditions Optimized</p>
                <p className="text-2xl font-mono font-bold">{synthesisData?.optimizerStats?.totalOptimized ?? 0}</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="border-[hsl(var(--gold)/0.2)]">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Layers className="h-5 w-5 text-[hsl(var(--gold))]" />
                  Parameter Space Categories
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {(synthesisData?.parameterSpace?.categories ?? []).map((cat, i) => (
                    <div key={i} className="p-3 bg-muted/30 rounded-md">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium">{cat.name}</span>
                        <Badge variant="secondary" className="text-xs">{cat.count} params</Badge>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {cat.parameters.map((p, j) => (
                          <span key={j} className="text-[10px] px-1.5 py-0.5 bg-background rounded border text-muted-foreground">
                            {p}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {synthesisData?.optimizerStats && synthesisData.optimizerStats.totalOptimized > 0 && (
              <Card className="border-[hsl(var(--gold)/0.2)]">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Target className="h-5 w-5 text-[hsl(var(--gold))]" />
                    Optimizer Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-2 bg-muted/30 rounded-md">
                        <p className="text-[10px] text-muted-foreground uppercase">Avg Feasibility</p>
                        <p className="text-lg font-mono font-bold">{(synthesisData.optimizerStats.avgFeasibility * 100).toFixed(1)}%</p>
                      </div>
                      <div className="p-2 bg-muted/30 rounded-md">
                        <p className="text-[10px] text-muted-foreground uppercase">Total Optimized</p>
                        <p className="text-lg font-mono font-bold">{synthesisData.optimizerStats.totalOptimized}</p>
                      </div>
                    </div>
                    {Object.keys(synthesisData.optimizerStats.methodBreakdown).length > 0 && (
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase mb-1">Methods</p>
                        <div className="flex flex-wrap gap-1.5">
                          {Object.entries(synthesisData.optimizerStats.methodBreakdown)
                            .sort(([, a], [, b]) => b - a)
                            .map(([method, count]) => (
                              <Badge key={method} variant="secondary" className="text-[10px] font-mono border-0">{method}: {count}</Badge>
                            ))}
                        </div>
                      </div>
                    )}
                    {synthesisData.optimizerStats.topConditions.length > 0 && (
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase mb-1">Top Conditions</p>
                        <div className="space-y-1">
                          {synthesisData.optimizerStats.topConditions.slice(0, 5).map((tc, i) => (
                            <div key={i} className="flex items-center justify-between text-xs bg-muted/30 px-2 py-1.5 rounded">
                              <span className="font-mono font-medium">{tc.formula}</span>
                              <div className="flex items-center gap-2 text-muted-foreground">
                                <span>{tc.method}</span>
                                <span className="font-mono">{(tc.feasibility * 100).toFixed(0)}%</span>
                                <span className="font-mono text-[hsl(var(--gold))]">{tc.tc}K</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="advanced-physics" className="space-y-4">
          <Card className="border-[hsl(var(--gold)/0.2)] p-6 text-center">
            <p className="text-muted-foreground">
              Advanced physics analysis runs automatically as the engine processes candidates.
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              View detailed physics properties, DFT results, and crystal structures in the{" "}
              <a href="/discovery-lab" className="text-[hsl(var(--gold))] hover:underline">Discovery Lab</a>.
            </p>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

