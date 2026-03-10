import { useQuery, useMutation } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useWebSocket } from "@/hooks/use-websocket";
import type { SuperconductorCandidate, ComputationalResult, CrystalStructure } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Link } from "wouter";
import {
  Atom, Zap, Filter, XCircle, CheckCircle2,
  Activity, Layers, Magnet, Gauge, Target,
  Beaker, ArrowDown, ExternalLink, Thermometer,
  FlaskConical, Star, Bug, Brain, Diamond, ClipboardList,
  Cpu, Clock, Loader2, Network, Code2, GitBranch, ArrowLeftRight,
  BookOpen, Sigma, FlaskRound, Search, TrendingUp, ShieldCheck, Lightbulb,
  GitMerge, ArrowRight, Play, Microscope, Settings2,
} from "lucide-react";
import DOSVisualizer from "@/components/dos-visualizer";

interface CalibrationResponse {
  r2: number;
  mae: number;
  rmse: number;
  absResidualPercentiles: { p50: number; p75: number; p90: number; p95: number };
  residualCount: number;
}

function computeConfidenceBand(predictedTc: number, p90: number): { lower: number; upper: number } {
  const scaleFactor = Math.max(1, predictedTc / 50);
  const errorMargin = p90 * Math.sqrt(scaleFactor);
  return {
    lower: Math.round(Math.max(0, predictedTc - errorMargin) * 10) / 10,
    upper: Math.round((predictedTc + errorMargin) * 10) / 10,
  };
}

const STAGE_NAMES = [
  "ML Filter",
  "Electronic Structure",
  "Phonon / E-Ph Coupling",
  "Tc Prediction (Eliashberg)",
  "Synthesis Feasibility",
];

const STAGE_COLORS = [
  "bg-gray-500",
  "bg-blue-500",
  "bg-purple-500",
  "bg-amber-500",
  "bg-green-500",
];

function PipelineFunnel({ stages }: { stages: { stage: number; count: number; passed: number }[] }) {
  const stageMap = new Map(stages.map(s => [s.stage, s]));

  return (
    <Card data-testid="pipeline-funnel">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Filter className="h-5 w-5" />
          Multi-Fidelity Screening Pipeline
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          5-stage computational pipeline: cheap models filter first, expensive methods confirm
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        {STAGE_NAMES.map((name, i) => {
          const data = stageMap.get(i);
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
                    <div
                      className="absolute inset-y-0 bg-red-500/30"
                      style={{
                        left: `${(passed / Math.max(total, 1)) * 100}%`,
                        width: `${(failed / Math.max(total, 1)) * 100}%`,
                      }}
                    />
                  </>
                )}
                <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono font-bold text-white mix-blend-difference">
                  {total > 0 ? `${passed}/${total}` : "awaiting candidates"}
                </div>
              </div>
              {i < 4 && (
                <div className="flex justify-center">
                  <ArrowDown className="h-3 w-3 text-muted-foreground" />
                </div>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function PhysicsPropertyCard({ candidate, p90 }: { candidate: SuperconductorCandidate; p90?: number }) {
  const competingPhases = (candidate.competingPhases as any[]) ?? [];
  const tcBand = candidate.predictedTc != null && p90 != null
    ? computeConfidenceBand(candidate.predictedTc, p90)
    : null;

  return (
    <Card data-testid={`physics-card-${candidate.id}`}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-base">{candidate.name}</CardTitle>
            <p className="text-sm font-mono text-primary">{candidate.formula}</p>
          </div>
          <div className="flex items-center gap-1">
            {candidate.verificationStage != null && (
              <Badge variant="outline" className="text-[10px]">Stage {candidate.verificationStage}/4</Badge>
            )}
            {candidate.dimensionality && (
              <Badge variant="outline" className="text-[10px]">{candidate.dimensionality}</Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {candidate.predictedTc != null && tcBand && (
          <div className="p-2 bg-muted/50 rounded-md" data-testid="tc-error-band">
            <div className="flex items-center gap-1.5 mb-1">
              <Thermometer className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Tc Prediction with 90% CI</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-muted-foreground">{tcBand.lower}K</span>
              <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden relative">
                <div
                  className="absolute inset-y-0 bg-primary/20 rounded-full"
                  style={{
                    left: `${(tcBand.lower / Math.max(tcBand.upper, 1)) * 100}%`,
                    right: "0%",
                  }}
                />
                <div
                  className="absolute inset-y-0 w-0.5 bg-primary"
                  style={{
                    left: `${(candidate.predictedTc / Math.max(tcBand.upper, 1)) * 100}%`,
                  }}
                />
              </div>
              <span className="text-xs font-mono text-muted-foreground">{tcBand.upper}K</span>
              <span className="text-xs font-mono font-bold">{candidate.predictedTc}K</span>
            </div>
          </div>
        )}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <PhysicsValue icon={Activity} label="e-ph lambda" value={candidate.electronPhononCoupling} format={(v) => v.toFixed(3)} />
          <PhysicsValue icon={Gauge} label="omega_log" value={candidate.logPhononFrequency} format={(v) => `${v.toFixed(0)} cm-1`} />
          <PhysicsValue icon={Atom} label="mu*" value={candidate.coulombPseudopotential} format={(v) => v.toFixed(3)} />
          <PhysicsValue icon={Target} label="U/W" value={candidate.correlationStrength} format={(v) => v.toFixed(3)} />
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <PhysicsValue icon={Magnet} label="Hc2" value={candidate.upperCriticalField} format={(v) => `${v.toFixed(1)} T`} />
          <PhysicsValue icon={Layers} label="xi" value={candidate.coherenceLength} format={(v) => `${v.toFixed(1)} nm`} />
          <PhysicsValue icon={Activity} label="lambda_L" value={candidate.londonPenetrationDepth} format={(v) => `${v.toFixed(0)} nm`} />
          <PhysicsValue icon={Zap} label="Jc" value={candidate.criticalCurrentDensity} format={(v) => `${(v / 1000).toFixed(0)}k A/cm2`} />
        </div>

        {(candidate.pairingMechanism || candidate.pairingSymmetry) && (
          <div className="flex flex-wrap gap-1.5">
            {candidate.pairingMechanism && <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 border-0 text-[10px]">{candidate.pairingMechanism}</Badge>}
            {candidate.pairingSymmetry && <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300 border-0 text-[10px]">{candidate.pairingSymmetry}</Badge>}
          </div>
        )}

        {candidate.fermiSurfaceTopology && (
          <div className="text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">Fermi Surface: </span>
            {candidate.fermiSurfaceTopology}
          </div>
        )}

        {competingPhases.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Competing Phases</p>
            <div className="space-y-1">
              {competingPhases.map((p: any, i: number) => (
                <div key={i} className="flex items-center justify-between text-xs p-1.5 bg-muted/40 rounded">
                  <div className="flex items-center gap-1.5">
                    {p.suppressesSC ? (
                      <XCircle className="h-3 w-3 text-red-500" />
                    ) : (
                      <CheckCircle2 className="h-3 w-3 text-green-500" />
                    )}
                    <span>{p.phaseName}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px]">{p.type}</Badge>
                    {p.transitionTemp && <span className="font-mono text-[10px]">{p.transitionTemp}K</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {candidate.uncertaintyEstimate != null && (
          <div className="flex items-center gap-2 text-xs border-t border-border pt-2">
            <span className="text-muted-foreground">Prediction Uncertainty:</span>
            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${candidate.uncertaintyEstimate > 0.6 ? "bg-red-500" : candidate.uncertaintyEstimate > 0.3 ? "bg-yellow-500" : "bg-green-500"}`}
                style={{ width: `${(candidate.uncertaintyEstimate ?? 0) * 100}%` }}
              />
            </div>
            <span className="font-mono font-bold">{((candidate.uncertaintyEstimate ?? 0) * 100).toFixed(0)}%</span>
          </div>
        )}

        <Link href={`/candidate/${encodeURIComponent(candidate.formula)}`} data-testid={`link-physics-profile-${candidate.id}`}>
          <span className="inline-flex items-center gap-1 text-xs text-primary hover:underline cursor-pointer mt-1">
            <ExternalLink className="h-3 w-3" />
            View Full Profile
          </span>
        </Link>
      </CardContent>
    </Card>
  );
}

function PhysicsValue({
  icon: Icon,
  label,
  value,
  format,
}: {
  icon: typeof Activity;
  label: string;
  value: number | null | undefined;
  format: (v: number) => string;
}) {
  if (value == null || !Number.isFinite(value)) return null;
  return (
    <div className="p-2 bg-muted/50 rounded-md">
      <div className="flex items-center gap-1 mb-0.5">
        <Icon className="h-3 w-3 text-muted-foreground" />
        <span className="text-muted-foreground text-[10px]">{label}</span>
      </div>
      <span className="font-mono font-bold text-xs">{format(value)}</span>
    </div>
  );
}

function FailedResultCard({ result }: { result: ComputationalResult }) {
  const results = result.results as Record<string, any>;

  return (
    <Card data-testid={`failed-result-${result.id}`} className="border-red-200 dark:border-red-900">
      <CardContent className="pt-4 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-mono text-sm font-bold">{result.formula}</p>
            <p className="text-xs text-muted-foreground">{result.computationType}</p>
          </div>
          <Badge variant="destructive" className="text-[10px]">Stage {result.pipelineStage}</Badge>
        </div>
        {result.failureReason && (
          <div className="flex items-start gap-2 text-xs p-2 bg-red-50 dark:bg-red-950/20 rounded">
            <XCircle className="h-3.5 w-3.5 text-red-500 mt-0.5 shrink-0" />
            <span className="text-red-700 dark:text-red-400">{result.failureReason}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function CrystalStructureCard({ structure }: { structure: CrystalStructure }) {
  const lattice = structure.latticeParams as Record<string, number> | null;

  return (
    <Card data-testid={`crystal-${structure.id}`}>
      <CardContent className="pt-4 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-mono text-sm font-bold">{structure.formula}</p>
            <p className="text-xs text-muted-foreground">{structure.spaceGroup} ({structure.crystalSystem})</p>
          </div>
          <div className="flex gap-1">
            {structure.isStable && <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">Stable</Badge>}
            {structure.isMetastable && <Badge className="bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300 border-0 text-[10px]">Metastable</Badge>}
            <Badge variant="outline" className="text-[10px]">{structure.dimensionality}</Badge>
          </div>
        </div>

        {structure.prototype && (
          <p className="text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">Prototype: </span>{structure.prototype}
          </p>
        )}

        {lattice && (
          <div className="grid grid-cols-3 gap-1 text-[10px]">
            <div className="p-1 bg-muted/50 rounded text-center">
              <span className="text-muted-foreground">a=</span>
              <span className="font-mono font-bold">{Number.isFinite(lattice.a) ? lattice.a.toFixed(2) : "--"}</span>
            </div>
            <div className="p-1 bg-muted/50 rounded text-center">
              <span className="text-muted-foreground">b=</span>
              <span className="font-mono font-bold">{Number.isFinite(lattice.b) ? lattice.b.toFixed(2) : "--"}</span>
            </div>
            <div className="p-1 bg-muted/50 rounded text-center">
              <span className="text-muted-foreground">c=</span>
              <span className="font-mono font-bold">{Number.isFinite(lattice.c) ? lattice.c.toFixed(2) : "--"}</span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 text-xs">
          {structure.convexHullDistance != null && Number.isFinite(structure.convexHullDistance) && (
            <div>
              <span className="text-muted-foreground">Hull dist: </span>
              <span className="font-mono">{structure.convexHullDistance.toFixed(3)} eV/atom</span>
            </div>
          )}
          {structure.synthesizability != null && Number.isFinite(structure.synthesizability) && (
            <div>
              <span className="text-muted-foreground">Synth: </span>
              <span className="font-mono">{(structure.synthesizability * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>

        <Link href={`/candidate/${encodeURIComponent(structure.formula)}`} data-testid={`link-crystal-profile-${structure.id}`}>
          <span className="inline-flex items-center gap-1 text-xs text-primary hover:underline cursor-pointer mt-1">
            <ExternalLink className="h-3 w-3" />
            View Full Profile
          </span>
        </Link>
      </CardContent>
    </Card>
  );
}

function NextGenPipelinePanel() {
  const { data: stats, isLoading } = useQuery<{
    activePipelines: number;
    totalRuns: number;
    totalCandidatesProcessed: number;
    pipelines: { id: string; status: string; iteration: number; bestTc: number; bestDistance: number; bestFormula: string }[];
  }>({ queryKey: ["/api/next-gen-pipeline/stats"], refetchInterval: 8000 });

  const defaultPipelineId = stats?.pipelines?.[0]?.id;

  const { data: detail } = useQuery<{
    status: string;
    iteration: number;
    bestTc: number;
    bestFormula: string;
    bestDistance: number;
    totalGenerated: number;
    totalPassed: number;
    convergenceHistory: number[];
    surrogateAccuracy: number;
    constraintPassRate: number;
    topCandidates: { formula: string; tc: number; distance: number; source: string }[];
    iterationsPerMinute: number;
    estimatedIterationsToConverge: number | null;
  }>({
    queryKey: ["/api/next-gen-pipeline", defaultPipelineId],
    enabled: !!defaultPipelineId,
    refetchInterval: 5000,
  });

  const createMutation = useMutation({
    mutationFn: async () => {
      const id = `pipeline-${Date.now()}`;
      await apiRequest("POST", "/api/next-gen-pipeline", { id, goal: { targetTc: 293, maxPressure: 50, minLambda: 1.5, maxHullDistance: 0.05, metallicRequired: true, phononStable: true, maxIterations: 200, convergenceThreshold: 0.02, surrogateWeight: 0.7, constraintStrictness: "standard" } });
    },
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/next-gen-pipeline/stats"] }); },
  });

  const iterateMutation = useMutation({
    mutationFn: async () => {
      if (!defaultPipelineId) return;
      await apiRequest("POST", `/api/next-gen-pipeline/${defaultPipelineId}/iterate`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/next-gen-pipeline/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/next-gen-pipeline", defaultPipelineId] });
    },
  });

  const togglePauseMutation = useMutation({
    mutationFn: async () => {
      if (!defaultPipelineId || !detail) return;
      const endpoint = detail.status === "paused" ? "resume" : "pause";
      await apiRequest("POST", `/api/next-gen-pipeline/${defaultPipelineId}/${endpoint}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/next-gen-pipeline/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/next-gen-pipeline", defaultPipelineId] });
    },
  });

  if (isLoading) return <Skeleton className="h-64" />;

  const stageFlow = [
    { label: "Goal Spec", icon: Target, color: "text-blue-500", desc: `Tc ${"\u2265"} 293K, P < 50 GPa` },
    { label: "Design Generator", icon: Brain, color: "text-purple-500", desc: `${detail?.totalGenerated ?? 0} candidates` },
    { label: "Constraint Solver", icon: Filter, color: "text-amber-500", desc: `${Math.round((detail?.constraintPassRate ?? 0) * 100)}% pass rate` },
    { label: "Surrogate Model", icon: Cpu, color: "text-green-500", desc: `${Math.round((detail?.surrogateAccuracy ?? 0) * 100)}% accuracy` },
    { label: "Learning Loop", icon: Activity, color: "text-red-500", desc: `${detail?.iteration ?? 0} iterations` },
  ];

  return (
    <div className="space-y-4">
      <Card data-testid="card-nextgen-overview">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-500" />
              Next-Generation Inverse Design Pipeline
            </CardTitle>
            <div className="flex gap-2">
              {!defaultPipelineId && (
                <Button size="sm" onClick={() => createMutation.mutate()} disabled={createMutation.isPending} className="bg-purple-500 hover:bg-purple-600 text-white text-xs" data-testid="btn-create-pipeline">
                  {createMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : null}
                  Initialize Pipeline
                </Button>
              )}
              {defaultPipelineId && detail?.status !== "converged" && detail?.status !== "completed" && (
                <>
                  <Button size="sm" onClick={() => iterateMutation.mutate()} disabled={iterateMutation.isPending || detail?.status === "paused"} className="bg-green-500 hover:bg-green-600 text-white text-xs" data-testid="btn-run-iteration">
                    {iterateMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : null}
                    Run Iteration
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => togglePauseMutation.mutate()} disabled={togglePauseMutation.isPending} data-testid="btn-toggle-pause">
                    {detail?.status === "paused" ? "Resume" : "Pause"}
                  </Button>
                </>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between gap-1 mb-4" data-testid="pipeline-flow">
            {stageFlow.map((stage, i) => (
              <div key={stage.label} className="flex items-center gap-1">
                <div className="flex flex-col items-center text-center min-w-[80px]">
                  <div className={`w-10 h-10 rounded-full bg-muted/50 border border-border flex items-center justify-center ${stage.color}`}>
                    <stage.icon className="h-5 w-5" />
                  </div>
                  <p className="text-[10px] font-medium mt-1">{stage.label}</p>
                  <p className="text-[9px] text-muted-foreground">{stage.desc}</p>
                </div>
                {i < stageFlow.length - 1 && (
                  <ArrowDown className="h-3 w-3 text-muted-foreground rotate-[-90deg] flex-shrink-0" />
                )}
              </div>
            ))}
          </div>

          {detail && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3" data-testid="pipeline-metrics">
              <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Status</p>
                <Badge variant={detail.status === "running" ? "default" : detail.status === "converged" ? "secondary" : "outline"} className="mt-1" data-testid="badge-pipeline-status">
                  {detail.status}
                </Badge>
              </div>
              <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Tc Predicted</p>
                <p className="text-lg font-mono font-bold text-foreground" data-testid="text-pipeline-best-tc">
                  {detail.bestTc > 0 ? `${Math.round(detail.bestTc)}K` : "--"}
                </p>
              </div>
              <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Target Distance</p>
                <p className="text-lg font-mono font-bold text-foreground" data-testid="text-pipeline-distance">
                  {detail.bestDistance < 1 ? detail.bestDistance.toFixed(3) : "--"}
                </p>
              </div>
              <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Iterations / min</p>
                <p className="text-lg font-mono font-bold text-foreground" data-testid="text-pipeline-speed">
                  {detail.iterationsPerMinute > 0 ? detail.iterationsPerMinute : "--"}
                </p>
              </div>
            </div>
          )}

          {!detail && !defaultPipelineId && (
            <div className="text-center py-8 text-muted-foreground" data-testid="pipeline-empty-state">
              <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No pipeline active. Initialize one to begin inverse design optimization.</p>
              <p className="text-xs mt-1">Goal: Tc {"\u2265"} 293K, zero resistance, Meissner effect, P {"<"} 50 GPa</p>
            </div>
          )}
        </CardContent>
      </Card>

      {detail && detail.convergenceHistory.length > 0 && (
        <Card data-testid="card-convergence">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              Convergence Trajectory
              {detail.estimatedIterationsToConverge !== null && (
                <span className="text-xs text-muted-foreground font-normal ml-2">
                  Est. {detail.estimatedIterationsToConverge} iterations to converge
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-24 flex items-end gap-[2px]" data-testid="convergence-chart">
              {detail.convergenceHistory.slice(-60).map((d, i) => {
                const maxD = Math.max(...detail.convergenceHistory.slice(-60), 0.001);
                const h = Math.max(2, (d / maxD) * 96);
                return (
                  <div key={i} className="flex-1 bg-green-500/60 rounded-t-sm transition-all" style={{ height: `${h}px` }} title={`Iter ${i + 1}: dist=${d.toFixed(4)}`} />
                );
              })}
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
              <span>Target distance (lower = closer to goal)</span>
              <span>Current: {detail.bestDistance.toFixed(4)}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {detail && detail.topCandidates.length > 0 && (
        <Card data-testid="card-top-candidates">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Star className="h-5 w-5 text-amber-500" />
              Top Pipeline Candidates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {detail.topCandidates.slice(0, 8).map((c, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-muted/30 rounded-md border border-border" data-testid={`candidate-row-${i}`}>
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-muted-foreground w-5">#{i + 1}</span>
                    <Link href={`/candidate/${encodeURIComponent(c.formula)}`}>
                      <span className="text-sm font-mono font-medium text-foreground hover:text-primary cursor-pointer" data-testid={`text-candidate-formula-${i}`}>
                        {c.formula}
                      </span>
                    </Link>
                    <Badge variant="outline" className="text-[10px]" data-testid={`badge-source-${i}`}>{c.source}</Badge>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="font-mono" data-testid={`text-candidate-tc-${i}`}>
                      {Math.round(c.tc)}K
                    </span>
                    <span className="text-muted-foreground font-mono" data-testid={`text-candidate-dist-${i}`}>
                      d={c.distance.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card data-testid="card-pipeline-architecture">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="h-5 w-5 text-blue-500" />
            5-Component Architecture
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {[
              { name: "Goal Specification", desc: "SC criteria: Tc, resistance, Meissner, pressure constraints", color: "border-blue-500/30 bg-blue-500/5" },
              { name: "Design Generator", desc: "Composition generation from learning bias + gradient refinement", color: "border-purple-500/30 bg-purple-500/5" },
              { name: "Constraint Solver", desc: "Physics constraints, 8-pillar SC evaluation, feasibility", color: "border-amber-500/30 bg-amber-500/5" },
              { name: "Surrogate Model", desc: "GB + GNN ensemble Tc prediction with uncertainty", color: "border-green-500/30 bg-green-500/5" },
              { name: "Learning Loop", desc: "Reward-driven bias updates, convergence tracking", color: "border-red-500/30 bg-red-500/5" },
            ].map((comp, i) => (
              <div key={i} className={`p-3 rounded-lg border ${comp.color}`} data-testid={`arch-component-${i}`}>
                <p className="text-xs font-semibold">{comp.name}</p>
                <p className="text-[10px] text-muted-foreground mt-1">{comp.desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function SelfImprovingLabPanel() {
  const { data: overview, isLoading } = useQuery<{
    activeLabs: number;
    totalRuns: number;
    labs: { id: string; status: string; iteration: number; bestTc: number; bestFormula: string; strategiesEvolved: number; knowledgeEntries: number }[];
  }>({ queryKey: ["/api/self-improving-lab/stats"], refetchInterval: 8000 });

  const defaultLabId = overview?.labs?.[0]?.id;

  const { data: detail } = useQuery<{
    id: string;
    status: string;
    iteration: number;
    bestTc: number;
    bestFormula: string;
    bestDistance: number;
    totalGenerated: number;
    totalPassed: number;
    activeStrategy: { id: string; name: string; type: string; fitness: number; bestTc: number } | null;
    strategies: { id: string; name: string; type: string; fitness: number; uses: number; bestTc: number; generation: number }[];
    knowledgeBaseSize: number;
    failureBreakdown: Record<string, number>;
    topKnowledge: { pattern: string; suggestion: string; confidence: number }[];
    convergenceHistory: number[];
    topCandidates: { formula: string; tc: number; distance: number; strategy: string }[];
    strategiesEvolved: number;
    iterationsPerMinute: number;
  }>({
    queryKey: ["/api/self-improving-lab", defaultLabId],
    enabled: !!defaultLabId,
    refetchInterval: 5000,
  });

  const createMutation = useMutation({
    mutationFn: async () => {
      const id = `lab-${Date.now()}`;
      await apiRequest("POST", "/api/self-improving-lab", { id, targetTc: 293, maxPressure: 50, maxIterations: 500 });
    },
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/self-improving-lab/stats"] }); },
  });

  const iterateMutation = useMutation({
    mutationFn: async () => {
      if (!defaultLabId) return;
      await apiRequest("POST", `/api/self-improving-lab/${defaultLabId}/iterate`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/self-improving-lab/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/self-improving-lab", defaultLabId] });
    },
  });

  const togglePauseMutation = useMutation({
    mutationFn: async () => {
      if (!defaultLabId || !detail) return;
      const endpoint = detail.status === "paused" ? "resume" : "pause";
      await apiRequest("POST", `/api/self-improving-lab/${defaultLabId}/${endpoint}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/self-improving-lab/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/self-improving-lab", defaultLabId] });
    },
  });

  if (isLoading) return <Skeleton className="h-64" />;

  const STRATEGY_COLORS: Record<string, string> = {
    "hydride-cage-optimizer": "text-blue-500 bg-blue-500/10 border-blue-500/30",
    "layered-intercalation": "text-purple-500 bg-purple-500/10 border-purple-500/30",
    "high-entropy-alloy": "text-green-500 bg-green-500/10 border-green-500/30",
    "light-element-phonon": "text-amber-500 bg-amber-500/10 border-amber-500/30",
    "topological-edge": "text-cyan-500 bg-cyan-500/10 border-cyan-500/30",
    "pressure-stabilized": "text-red-500 bg-red-500/10 border-red-500/30",
    "electron-phonon-resonance": "text-pink-500 bg-pink-500/10 border-pink-500/30",
    "charge-transfer-layer": "text-orange-500 bg-orange-500/10 border-orange-500/30",
    custom: "text-gray-500 bg-gray-500/10 border-gray-500/30",
  };

  const FAILURE_LABELS: Record<string, string> = {
    "low-tc": "Low Tc",
    "constraint-violation": "Constraint Violation",
    "high-pressure": "High Pressure",
    "thermodynamic-instability": "Thermodynamic Instability",
    "poor-electron-phonon": "Weak E-Ph Coupling",
    "synthesis-infeasible": "Synthesis Infeasible",
    "phonon-instability": "Phonon Instability",
    "insufficient-dos": "Insufficient DOS",
  };

  return (
    <div className="space-y-4">
      <Card data-testid="card-lab-overview">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-500" />
              Self-Improving Design Lab
            </CardTitle>
            <div className="flex gap-2">
              {!defaultLabId && (
                <Button size="sm" onClick={() => createMutation.mutate()} disabled={createMutation.isPending} className="bg-purple-500 hover:bg-purple-600 text-white text-xs" data-testid="btn-create-lab">
                  {createMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : null}
                  Initialize Lab
                </Button>
              )}
              {defaultLabId && detail?.status !== "converged" && detail?.status !== "completed" && (
                <>
                  <Button size="sm" onClick={() => iterateMutation.mutate()} disabled={iterateMutation.isPending || detail?.status === "paused"} className="bg-green-500 hover:bg-green-600 text-white text-xs" data-testid="btn-run-lab-iteration">
                    {iterateMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : null}
                    Run Iteration
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => togglePauseMutation.mutate()} disabled={togglePauseMutation.isPending} data-testid="btn-toggle-lab-pause">
                    {detail?.status === "paused" ? "Resume" : "Pause"}
                  </Button>
                </>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {detail ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3" data-testid="lab-metrics">
                <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Status</p>
                  <Badge variant={detail.status === "running" ? "default" : detail.status === "converged" ? "secondary" : "outline"} className="mt-1" data-testid="badge-lab-status">
                    {detail.status}
                  </Badge>
                </div>
                <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Tc Predicted</p>
                  <p className="text-lg font-mono font-bold text-foreground" data-testid="text-lab-best-tc">
                    {detail.bestTc > 0 ? `${Math.round(detail.bestTc)}K` : "--"}
                  </p>
                </div>
                <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Active Strategy</p>
                  <p className="text-sm font-medium truncate" data-testid="text-lab-active-strategy">
                    {detail.activeStrategy?.name ?? "--"}
                  </p>
                </div>
                <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Strategies Evolved</p>
                  <p className="text-lg font-mono font-bold text-foreground" data-testid="text-lab-evolved">
                    {detail.strategiesEvolved}
                  </p>
                </div>
                <div className="p-2.5 bg-muted/30 rounded-lg border border-border">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Knowledge Entries</p>
                  <p className="text-lg font-mono font-bold text-foreground" data-testid="text-lab-knowledge">
                    {detail.knowledgeBaseSize}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1 overflow-x-auto" data-testid="lab-pipeline-flow">
                {[
                  { label: "Goal Spec", icon: Target, color: "text-blue-500" },
                  { label: "Strategy Gen", icon: Brain, color: "text-purple-500" },
                  { label: "Design Gen", icon: Cpu, color: "text-green-500" },
                  { label: "Constraints", icon: Filter, color: "text-amber-500" },
                  { label: "Surrogate", icon: Gauge, color: "text-cyan-500" },
                  { label: "INR Field", icon: Diamond, color: "text-pink-500" },
                  { label: "Evaluation", icon: CheckCircle2, color: "text-green-500" },
                  { label: "Failure Analysis", icon: Bug, color: "text-red-500" },
                  { label: "Strategy Evolve", icon: Activity, color: "text-orange-500" },
                ].map((stage, i) => (
                  <div key={stage.label} className="flex items-center gap-1 flex-shrink-0">
                    <div className="flex flex-col items-center text-center min-w-[62px]">
                      <div className={`w-8 h-8 rounded-full bg-muted/50 border border-border flex items-center justify-center ${stage.color}`}>
                        <stage.icon className="h-4 w-4" />
                      </div>
                      <p className="text-[9px] font-medium mt-0.5">{stage.label}</p>
                    </div>
                    {i < 8 && <ArrowDown className="h-2.5 w-2.5 text-muted-foreground rotate-[-90deg] flex-shrink-0" />}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground" data-testid="lab-empty-state">
              <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No design lab active. Initialize one to begin self-improving material design.</p>
              <p className="text-xs mt-1">The lab learns design strategies, not just parameters -- evolving its approach over time.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {detail && detail.strategies.length > 0 && (
        <Card data-testid="card-strategy-pool">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Layers className="h-5 w-5 text-blue-500" />
              Strategy Pool ({detail.strategies.length} active)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
              {detail.strategies
                .sort((a, b) => b.fitness - a.fitness)
                .map((s, i) => {
                  const colors = STRATEGY_COLORS[s.type] ?? STRATEGY_COLORS.custom;
                  const isActive = s.id === detail.activeStrategy?.id;
                  return (
                    <div key={i} className={`p-2.5 rounded-lg border ${colors} ${isActive ? "ring-1 ring-primary" : ""}`} data-testid={`strategy-card-${i}`}>
                      <div className="flex items-center justify-between mb-1">
                        <p className="text-xs font-semibold truncate">{s.name}</p>
                        {isActive && <Badge variant="default" className="text-[8px] px-1 py-0">Active</Badge>}
                      </div>
                      <div className="grid grid-cols-3 gap-1 text-[10px]">
                        <div>
                          <span className="text-muted-foreground">Fit</span>
                          <p className="font-mono font-bold">{s.fitness.toFixed(2)}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Best</span>
                          <p className="font-mono font-bold">{s.bestTc > 0 ? `${s.bestTc}K` : "--"}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Gen</span>
                          <p className="font-mono font-bold">{s.generation}</p>
                        </div>
                      </div>
                      <p className="text-[9px] text-muted-foreground mt-1">{s.uses} uses</p>
                    </div>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      )}

      {detail && Object.keys(detail.failureBreakdown).length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card data-testid="card-failure-analysis">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Bug className="h-5 w-5 text-red-500" />
                Failure Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(detail.failureBreakdown)
                  .sort(([, a], [, b]) => b - a)
                  .map(([type, count], i) => {
                    const total = Object.values(detail.failureBreakdown).reduce((s, v) => s + v, 0);
                    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                    return (
                      <div key={i} className="flex items-center gap-2" data-testid={`failure-row-${i}`}>
                        <span className="text-xs w-36 truncate">{FAILURE_LABELS[type] ?? type}</span>
                        <div className="flex-1 h-4 bg-muted/30 rounded-full overflow-hidden">
                          <div className="h-full bg-red-500/50 rounded-full" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-xs font-mono text-muted-foreground w-12 text-right">{count}</span>
                      </div>
                    );
                  })}
              </div>
            </CardContent>
          </Card>

          <Card data-testid="card-knowledge-base">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <ClipboardList className="h-5 w-5 text-amber-500" />
                Knowledge Base
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {detail.topKnowledge.map((k, i) => (
                  <div key={i} className="p-2 bg-muted/30 rounded-md border border-border" data-testid={`knowledge-entry-${i}`}>
                    <div className="flex items-center justify-between mb-1">
                      <Badge variant="outline" className="text-[9px]">{k.pattern}</Badge>
                      <span className="text-[10px] text-muted-foreground">{Math.round(k.confidence * 100)}% conf</span>
                    </div>
                    <p className="text-[10px] text-muted-foreground">{k.suggestion}</p>
                  </div>
                ))}
                {detail.topKnowledge.length === 0 && (
                  <p className="text-xs text-muted-foreground text-center py-4">No knowledge entries yet. Run iterations to build the knowledge base.</p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {detail && detail.convergenceHistory.length > 0 && (
        <Card data-testid="card-lab-convergence">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              Convergence Trajectory
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-24 flex items-end gap-[2px]" data-testid="lab-convergence-chart">
              {detail.convergenceHistory.slice(-80).map((d, i) => {
                const maxD = Math.max(...detail.convergenceHistory.slice(-80), 0.001);
                const h = Math.max(2, (d / maxD) * 96);
                return (
                  <div key={i} className="flex-1 bg-purple-500/60 rounded-t-sm transition-all" style={{ height: `${h}px` }} title={`Iter ${i + 1}: dist=${d.toFixed(4)}`} />
                );
              })}
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
              <span>Target distance (lower = closer to goal)</span>
              <span>Best: {detail.bestDistance.toFixed(4)}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {detail && detail.topCandidates.length > 0 && (
        <Card data-testid="card-lab-candidates">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Star className="h-5 w-5 text-amber-500" />
              Top Lab Candidates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {detail.topCandidates.slice(0, 8).map((c, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-muted/30 rounded-md border border-border" data-testid={`lab-candidate-row-${i}`}>
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-muted-foreground w-5">#{i + 1}</span>
                    <Link href={`/candidate/${encodeURIComponent(c.formula)}`}>
                      <span className="text-sm font-mono font-medium text-foreground hover:text-primary cursor-pointer" data-testid={`text-lab-formula-${i}`}>
                        {c.formula}
                      </span>
                    </Link>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="font-mono" data-testid={`text-lab-tc-${i}`}>
                      {Math.round(c.tc)}K
                    </span>
                    <span className="text-muted-foreground font-mono">
                      d={c.distance.toFixed(3)}
                    </span>
                    <Badge variant="outline" className="text-[9px]">{c.strategy}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card data-testid="card-lab-architecture">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="h-5 w-5 text-blue-500" />
            Self-Improving Architecture
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { name: "Strategy-Level Optimization", desc: "Evolves design strategies (architectures), not just parameters. 8 concurrent strategy types compete.", color: "border-purple-500/30 bg-purple-500/5" },
              { name: "Failure Analysis Engine", desc: "Classifies why designs fail (low Tc, constraint violations, instability) and generates corrective suggestions.", color: "border-red-500/30 bg-red-500/5" },
              { name: "Implicit Neural Representations", desc: "MLP(x,y,z) maps to material density, enabling infinite-resolution geometry with gradient and curvature.", color: "border-pink-500/30 bg-pink-500/5" },
              { name: "Knowledge Base", desc: "Stores failure patterns and suggestions with confidence scores. Drives strategy evolution.", color: "border-amber-500/30 bg-amber-500/5" },
              { name: "Strategy Evolution", desc: "Replaces low-performing strategies with evolved children. Uses crossover, mutation, and knowledge-guided adaptation.", color: "border-green-500/30 bg-green-500/5" },
              { name: "Surrogate + Real Physics", desc: "GB+GNN ensemble surrogate with 8-pillar SC evaluation. Physics constraint validation.", color: "border-cyan-500/30 bg-cyan-500/5" },
            ].map((comp, i) => (
              <div key={i} className={`p-3 rounded-lg border ${comp.color}`} data-testid={`lab-arch-component-${i}`}>
                <p className="text-xs font-semibold">{comp.name}</p>
                <p className="text-[10px] text-muted-foreground mt-1">{comp.desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function AdvancedPhysicsPanel() {
  const { data: defectStats, isLoading: defectLoading } = useQuery<{
    totalDefectsGenerated: number;
    typeBreakdown: Record<string, number>;
    bestTcImprovement: number;
    bestTcFormula: string;
    totalFormulasProcessed: number;
  }>({ queryKey: ["/api/defect-engine/stats"] });

  const { data: corrStats, isLoading: corrLoading } = useQuery<{
    materialsAnalyzed: number;
    regimeBreakdown: Record<string, number>;
    avgCorrelationScore: number;
  }>({ queryKey: ["/api/correlation-engine/stats"] });

  const { data: growthStats, isLoading: growthLoading } = useQuery<{
    totalSimulations: number;
    avgGrainSize: number;
    qualityDistribution: Record<string, number>;
    bestQualityScore: number;
    bestFormula: string;
  }>({ queryKey: ["/api/crystal-growth/stats"] });

  const { data: plannerStats, isLoading: plannerLoading } = useQuery<{
    totalPlansGenerated: number;
    totalMethodsSuggested: number;
    methodFrequency: Record<string, number>;
    topCandidates: { formula: string; score: number }[];
    avgExperimentScore: number;
  }>({ queryKey: ["/api/experiment-planner/stats"] });

  const { messages } = useWebSocket();
  useEffect(() => {
    const engineEvents = messages.filter(m => m.type === "log" && (m.data?.phase === "defect-engine" || m.data?.phase === "correlation-engine" || m.data?.phase === "crystal-growth"));
    if (engineEvents.length > 0) {
      queryClient.invalidateQueries({ queryKey: ["/api/defect-engine/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/correlation-engine/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/crystal-growth/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/experiment-planner/stats"] });
    }
  }, [messages]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card data-testid="card-defect-engine">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Bug className="h-5 w-5 text-orange-500" />
            Defect Physics Engine
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Vacancy, interstitial, antisite, and dopant defect modeling
          </p>
        </CardHeader>
        <CardContent>
          {defectLoading ? <Skeleton className="h-32" /> : defectStats ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-defects-generated">
                    {defectStats.totalDefectsGenerated}
                  </div>
                  <div className="text-xs text-muted-foreground">Defects Generated</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-formulas-processed">
                    {defectStats.totalFormulasProcessed}
                  </div>
                  <div className="text-xs text-muted-foreground">Formulas Processed</div>
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground">Type Breakdown</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(defectStats.typeBreakdown).map(([type, count]) => (
                    <Badge key={type} variant="outline" className="text-xs font-mono" data-testid={`badge-defect-${type}`}>
                      {type}: {count}
                    </Badge>
                  ))}
                </div>
              </div>
              {defectStats.bestTcFormula && (
                <div className="text-xs text-muted-foreground">
                  Best Tc improvement: <span className="font-mono text-foreground">{defectStats.bestTcImprovement.toFixed(1)}x</span>
                  {" "}on <Link href={`/candidate/${encodeURIComponent(defectStats.bestTcFormula)}`}><span className="text-primary hover:underline cursor-pointer">{defectStats.bestTcFormula}</span></Link>
                </div>
              )}
            </div>
          ) : <div className="text-sm text-muted-foreground">No data available</div>}
        </CardContent>
      </Card>

      <Card data-testid="card-correlation-engine">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-500" />
            Strong Correlation Engine
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Mott physics, spin fluctuations, and Kondo lattice detection
          </p>
        </CardHeader>
        <CardContent>
          {corrLoading ? <Skeleton className="h-32" /> : corrStats ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-materials-analyzed">
                    {corrStats.materialsAnalyzed}
                  </div>
                  <div className="text-xs text-muted-foreground">Materials Analyzed</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-avg-correlation">
                    {corrStats.avgCorrelationScore.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground">Avg Correlation Score</div>
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground">Regime Breakdown</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(corrStats.regimeBreakdown).filter(([, c]) => c > 0).map(([regime, count]) => (
                    <Badge key={regime} variant="outline" className="text-xs font-mono" data-testid={`badge-regime-${regime}`}>
                      {regime}: {count}
                    </Badge>
                  ))}
                  {Object.values(corrStats.regimeBreakdown).every(c => c === 0) && (
                    <span className="text-xs text-muted-foreground">Awaiting candidate analysis</span>
                  )}
                </div>
              </div>
            </div>
          ) : <div className="text-sm text-muted-foreground">No data available</div>}
        </CardContent>
      </Card>

      <Card data-testid="card-crystal-growth">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Diamond className="h-5 w-5 text-cyan-500" />
            Crystal Growth Simulator
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Nucleation, grain structure, and critical current modeling
          </p>
        </CardHeader>
        <CardContent>
          {growthLoading ? <Skeleton className="h-32" /> : growthStats ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-growth-simulations">
                    {growthStats.totalSimulations}
                  </div>
                  <div className="text-xs text-muted-foreground">Simulations Run</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-avg-grain-size">
                    {growthStats.avgGrainSize > 0 ? `${growthStats.avgGrainSize.toFixed(0)} nm` : "--"}
                  </div>
                  <div className="text-xs text-muted-foreground">Avg Grain Size</div>
                </div>
              </div>
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground">Quality Distribution</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(growthStats.qualityDistribution).map(([quality, count]) => {
                    const colors: Record<string, string> = {
                      poor: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
                      fair: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400",
                      good: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
                      excellent: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
                    };
                    return (
                      <span key={quality} className={`text-xs font-mono px-2 py-0.5 rounded ${colors[quality] || ""}`} data-testid={`badge-quality-${quality}`}>
                        {quality}: {count}
                      </span>
                    );
                  })}
                </div>
              </div>
              {growthStats.bestFormula && (
                <div className="text-xs text-muted-foreground">
                  Best quality: <span className="font-mono text-foreground">{growthStats.bestQualityScore.toFixed(2)}</span>
                  {" "}on <Link href={`/candidate/${encodeURIComponent(growthStats.bestFormula)}`}><span className="text-primary hover:underline cursor-pointer">{growthStats.bestFormula}</span></Link>
                </div>
              )}
            </div>
          ) : <div className="text-sm text-muted-foreground">No data available</div>}
        </CardContent>
      </Card>

      <Card data-testid="card-experiment-planner">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <ClipboardList className="h-5 w-5 text-green-500" />
            Experimental Validation Planner
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Lab instructions, characterization methods, and candidate ranking
          </p>
        </CardHeader>
        <CardContent>
          {plannerLoading ? <Skeleton className="h-32" /> : plannerStats ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-plans-generated">
                    {plannerStats.totalPlansGenerated}
                  </div>
                  <div className="text-xs text-muted-foreground">Plans Generated</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold font-mono" data-testid="text-methods-suggested">
                    {plannerStats.totalMethodsSuggested}
                  </div>
                  <div className="text-xs text-muted-foreground">Methods Suggested</div>
                </div>
              </div>
              {Object.keys(plannerStats.methodFrequency).length > 0 && (
                <div className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Top Characterization Methods</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(plannerStats.methodFrequency)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 6)
                      .map(([method, count]) => (
                        <Badge key={method} variant="outline" className="text-xs font-mono" data-testid={`badge-method-${method}`}>
                          {method}: {count}
                        </Badge>
                      ))}
                  </div>
                </div>
              )}
              {plannerStats.topCandidates.length > 0 && (
                <div className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Top Candidates for Experiment</div>
                  <div className="space-y-1">
                    {plannerStats.topCandidates.slice(0, 3).map((c, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <Link href={`/candidate/${encodeURIComponent(c.formula)}`}>
                          <span className="text-primary hover:underline cursor-pointer font-mono" data-testid={`link-candidate-${i}`}>{c.formula}</span>
                        </Link>
                        <span className="font-mono text-muted-foreground">score: {c.score.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {plannerStats.avgExperimentScore > 0 && (
                <div className="text-xs text-muted-foreground">
                  Avg experiment score: <span className="font-mono text-foreground">{plannerStats.avgExperimentScore.toFixed(2)}</span>
                </div>
              )}
            </div>
          ) : <div className="text-sm text-muted-foreground">No data available</div>}
        </CardContent>
      </Card>
    </div>
  );
}

interface DFTQueueJob {
  id: number;
  formula: string;
  status: string;
  jobType: string;
  priority: number;
  createdAt: string | null;
  startedAt: string | null;
  completedAt: string | null;
  wallTime: number | null;
  converged: boolean;
  totalEnergy: number | null;
  error: string | null;
}

interface DFTQueueStatsData {
  queued: number;
  running: number;
  completed: number;
  failed: number;
  totalProcessed: number;
  totalSucceeded: number;
  totalFailed: number;
  isProcessing: boolean;
  currentFormula: string | null;
  qeAvailable: boolean;
  recentJobs: DFTQueueJob[];
}

function DFTQueuePanel() {
  const { lastMessage } = useWebSocket();
  const { data: queueStats, isLoading } = useQuery<DFTQueueStatsData>({
    queryKey: ["/api/dft-queue/stats"],
    refetchInterval: 15000,
  });

  useEffect(() => {
    if (lastMessage?.type === "dftJobQueued" || lastMessage?.type === "dftJobCompleted" || lastMessage?.type === "dftJobStarted") {
      queryClient.invalidateQueries({ queryKey: ["/api/dft-queue/stats"] });
    }
  }, [lastMessage]);

  if (isLoading) return <Skeleton className="h-48 w-full" />;
  if (!queueStats) return null;

  const statusColors: Record<string, string> = {
    queued: "bg-blue-500/10 text-blue-700 border-blue-300",
    running: "bg-amber-500/10 text-amber-700 border-amber-300",
    completed: "bg-green-500/10 text-green-700 border-green-300",
    failed: "bg-red-500/10 text-red-700 border-red-300",
  };

  return (
    <Card data-testid="dft-queue-panel">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Cpu className="h-5 w-5 text-violet-500" />
          Quantum ESPRESSO Full DFT Queue
          {queueStats.qeAvailable && (
            <Badge variant="outline" className="ml-2 text-[10px] bg-green-500/10 text-green-700 border-green-300" data-testid="badge-qe-available">
              QE 7.2 Active
            </Badge>
          )}
          {queueStats.isProcessing && (
            <Badge variant="outline" className="ml-1 text-[10px] bg-amber-500/10 text-amber-700 border-amber-300 flex items-center gap-1" data-testid="badge-qe-processing">
              <Loader2 className="h-3 w-3 animate-spin" />
              Processing
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3" data-testid="dft-queue-counters">
          <div className="p-3 rounded-md bg-blue-500/10 border border-blue-200">
            <p className="text-xs text-muted-foreground">Queued</p>
            <p className="text-xl font-bold text-blue-700" data-testid="text-dft-queued">{queueStats.queued}</p>
          </div>
          <div className="p-3 rounded-md bg-amber-500/10 border border-amber-200">
            <p className="text-xs text-muted-foreground">Running</p>
            <p className="text-xl font-bold text-amber-700" data-testid="text-dft-running">{queueStats.running}</p>
          </div>
          <div className="p-3 rounded-md bg-green-500/10 border border-green-200">
            <p className="text-xs text-muted-foreground">Completed</p>
            <p className="text-xl font-bold text-green-700" data-testid="text-dft-completed">{queueStats.completed}</p>
          </div>
          <div className="p-3 rounded-md bg-red-500/10 border border-red-200">
            <p className="text-xs text-muted-foreground">Failed</p>
            <p className="text-xl font-bold text-red-700" data-testid="text-dft-failed">{queueStats.failed}</p>
          </div>
        </div>

        {queueStats.currentFormula && (
          <div className="p-3 rounded-md bg-amber-500/5 border border-amber-200 flex items-center gap-3" data-testid="dft-current-job">
            <Loader2 className="h-4 w-4 animate-spin text-amber-600" />
            <div>
              <p className="text-sm font-medium">Currently computing: {queueStats.currentFormula}</p>
              <p className="text-xs text-muted-foreground">PBE/SCF + Phonon via pw.x / ph.x</p>
            </div>
          </div>
        )}

        {queueStats.recentJobs.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-medium text-muted-foreground mb-2">Recent DFT Jobs (top 0.1% candidates)</p>
            <div className="max-h-48 overflow-y-auto space-y-1" data-testid="dft-recent-jobs">
              {queueStats.recentJobs.map((job) => (
                <div key={job.id} className="flex items-center justify-between p-2 rounded bg-muted/30 text-xs" data-testid={`dft-job-${job.id}`}>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className={`text-[9px] px-1.5 ${statusColors[job.status] || ""}`}>
                      {job.status}
                    </Badge>
                    <span className="font-mono font-medium">{job.formula}</span>
                  </div>
                  <div className="flex items-center gap-3 text-muted-foreground">
                    {job.converged && job.totalEnergy && (
                      <span>E={job.totalEnergy.toFixed(2)} eV</span>
                    )}
                    {job.wallTime !== null && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {job.wallTime.toFixed(1)}s
                      </span>
                    )}
                    {job.error && (
                      <span className="text-red-500 truncate max-w-32" title={job.error}>{job.error.slice(0, 40)}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function ComputationalPhysics() {
  const { data: scData, isLoading: scLoading } = useQuery<{ candidates: SuperconductorCandidate[]; total: number }>({
    queryKey: ["/api/superconductor-candidates", { limit: 100 }],
  });

  const { data: pipelineData } = useQuery<{ pipelineStages: { stage: number; count: number; passed: number }[]; crystalStructures: number; computationalResults: number }>({
    queryKey: ["/api/pipeline-stats"],
  });

  const { data: calibrationData } = useQuery<CalibrationResponse>({
    queryKey: ["/api/ml-calibration"],
  });

  const p90 = calibrationData?.absResidualPercentiles?.p90;

  const { data: failedData } = useQuery<{ results: ComputationalResult[] }>({
    queryKey: ["/api/computational-results/failed"],
  });

  const { data: structureData } = useQuery<{ structures: CrystalStructure[]; total: number }>({
    queryKey: ["/api/crystal-structures"],
  });

  const { data: dftStatus } = useQuery<{
    total: number;
    dftEnrichedCount: number;
    breakdown: { high: number; medium: number; analytical: number };
    recentEnriched: { formula: string; confidence: string; ensembleScore: number; predictedTc: number }[];
  }>({
    queryKey: ["/api/dft-status"],
  });

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
    refetchInterval: 30000,
  });

  const { data: simData } = useQuery<{
    simulator: {
      totalSimulations: number;
      totalMutations: number;
      totalPathsOptimized: number;
      avgTcImprovement: number;
      bestTcImprovement: number;
      bestFormula: string;
      feasibilityBreakdown: Record<string, number>;
      modeBreakdown: Record<string, number>;
    };
    learning: {
      totalRecords: number;
      uniqueFormulas: number;
      avgTc: number;
      bestTc: number;
      bestFormula: string;
      patterns: { description: string; confidence: number; sampleCount: number; avgTcImprovement: number }[];
      classBreakdown: Record<string, { count: number; avgTc: number; bestTc: number }>;
      parameterCorrelations: { parameter: string; correlation: number }[];
    };
  }>({
    queryKey: ["/api/synthesis-simulator/stats"],
    refetchInterval: 30000,
  });

  const ws = useWebSocket();

  useEffect(() => {
    const relevantTypes = ["phaseUpdate", "progress", "prediction", "insight", "cycleEnd", "log"];
    const hasRelevant = ws.messages.some((m) => relevantTypes.includes(m.type));
    if (hasRelevant) {
      queryClient.invalidateQueries({ queryKey: ["/api/superconductor-candidates"] });
      queryClient.invalidateQueries({ queryKey: ["/api/pipeline-stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/computational-results/failed"] });
      queryClient.invalidateQueries({ queryKey: ["/api/crystal-structures"] });
      queryClient.invalidateQueries({ queryKey: ["/api/synthesis-variables/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/synthesis-simulator/stats"] });
    }
  }, [ws.messages.length]);

  const candidates = scData?.candidates ?? [];
  const physicsAnalyzed = candidates.filter(c => c.electronPhononCoupling != null || c.verificationStage != null && c.verificationStage > 0);
  const failedResults = failedData?.results ?? [];
  const rawStructures = structureData?.structures ?? [];
  const structures = (() => {
    const byFormula = new Map<string, CrystalStructure>();
    for (const s of rawStructures) {
      const existing = byFormula.get(s.formula);
      if (!existing || (s.convexHullDistance != null && (existing.convexHullDistance == null || s.convexHullDistance < existing.convexHullDistance))) {
        byFormula.set(s.formula, s);
      }
    }
    return Array.from(byFormula.values());
  })();

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight" data-testid="text-page-title">Computational Physics Engine</h1>
        <p className="text-muted-foreground mt-1">
          DFT-informed electronic structure, phonon spectra, Eliashberg theory, competing phases, and multi-fidelity screening
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Physics Analyzed</p>
            <p className="text-2xl font-mono font-bold" data-testid="stat-physics-analyzed">{physicsAnalyzed.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Crystal Structures</p>
            <p className="text-2xl font-mono font-bold" data-testid="stat-crystal-structures">{pipelineData?.crystalStructures ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Computations Run</p>
            <p className="text-2xl font-mono font-bold" data-testid="stat-computations">{pipelineData?.computationalResults ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Failed Screenings</p>
            <p className="text-2xl font-mono font-bold text-red-500" data-testid="stat-failed">{failedResults.length}</p>
          </CardContent>
        </Card>
      </div>

      <DFTQueuePanel />

      <Tabs defaultValue="pipeline" className="space-y-4">
        <TabsList data-testid="physics-tabs">
          <TabsTrigger value="pipeline" data-testid="tab-pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="physics" data-testid="tab-physics">Physics Properties</TabsTrigger>
          <TabsTrigger value="dft-selections" data-testid="tab-dft-selections">DFT Selections</TabsTrigger>
          <TabsTrigger value="structures" data-testid="tab-structures">Crystal Structures</TabsTrigger>
          <TabsTrigger value="failures" data-testid="tab-failures">Negative Results</TabsTrigger>
          <TabsTrigger value="synthesis" data-testid="tab-synthesis">Synthesis Variables</TabsTrigger>
          <TabsTrigger value="advanced-physics" data-testid="tab-advanced-physics">Advanced Physics</TabsTrigger>
          <TabsTrigger value="next-gen-pipeline" data-testid="tab-next-gen-pipeline">Inverse Design</TabsTrigger>
          <TabsTrigger value="self-improving-lab" data-testid="tab-self-improving-lab">Design Lab</TabsTrigger>
          <TabsTrigger value="design-repr" data-testid="tab-design-repr">Representations</TabsTrigger>
          <TabsTrigger value="theory-discovery" data-testid="tab-theory-discovery">Theory Discovery</TabsTrigger>
          <TabsTrigger value="causal-discovery" data-testid="tab-causal-discovery">Causal Discovery</TabsTrigger>
          <TabsTrigger value="dos-surrogate" data-testid="tab-dos-surrogate">DOS Surrogate</TabsTrigger>
        </TabsList>

        <TabsContent value="pipeline" className="space-y-4">
          <PipelineFunnel stages={pipelineData?.pipelineStages ?? []} />
        </TabsContent>

        <TabsContent value="physics" className="space-y-4">
          {scLoading ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {[1,2,3,4].map(i => <Skeleton key={i} className="h-64" />)}
            </div>
          ) : physicsAnalyzed.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {physicsAnalyzed.map(c => <PhysicsPropertyCard key={c.id} candidate={c} p90={p90} />)}
            </div>
          ) : (
            <Card>
              <CardContent className="pt-8 pb-8 text-center text-muted-foreground">
                <Atom className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Physics engine will analyze candidates once SC research generates them</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="dft-selections" className="space-y-4" data-testid="dft-selections-content">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <FlaskConical className="h-5 w-5" />
                Uncertainty-Driven DFT Selections
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Candidates selected for DFT enrichment based on acquisition score (0.5 x Tc + 0.5 x uncertainty)
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <div className="p-2.5 bg-muted/50 rounded-md" data-testid="dft-high-confidence">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">High Confidence</p>
                  <p className="text-xl font-mono font-bold text-green-600 dark:text-green-400">{dftStatus?.breakdown?.high ?? 0}</p>
                </div>
                <div className="p-2.5 bg-muted/50 rounded-md" data-testid="dft-medium-confidence">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Medium Confidence</p>
                  <p className="text-xl font-mono font-bold text-yellow-600 dark:text-yellow-400">{dftStatus?.breakdown?.medium ?? 0}</p>
                </div>
                <div className="p-2.5 bg-muted/50 rounded-md" data-testid="dft-analytical">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Analytical Only</p>
                  <p className="text-xl font-mono font-bold text-muted-foreground">{dftStatus?.breakdown?.analytical ?? 0}</p>
                </div>
              </div>

              {(() => {
                const highUncertainty = candidates
                  .filter(c => c.uncertaintyEstimate != null && c.uncertaintyEstimate > 0.3)
                  .sort((a, b) => {
                    const aScore = 0.5 * Math.min(1, (a.predictedTc ?? 0) / 300) + 0.5 * (a.uncertaintyEstimate ?? 0);
                    const bScore = 0.5 * Math.min(1, (b.predictedTc ?? 0) / 300) + 0.5 * (b.uncertaintyEstimate ?? 0);
                    return bScore - aScore;
                  })
                  .slice(0, 10);

                if (highUncertainty.length === 0) {
                  return (
                    <div className="text-center py-6 text-muted-foreground text-sm" data-testid="no-uncertain-candidates">
                      <Target className="h-6 w-6 mx-auto mb-2 opacity-50" />
                      <p>No high-uncertainty candidates awaiting DFT enrichment</p>
                    </div>
                  );
                }

                return (
                  <div className="space-y-2" data-testid="uncertain-candidates-list">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Top Candidates for DFT (by acquisition score)</p>
                    {highUncertainty.map((c, i) => {
                      const acqScore = 0.5 * Math.min(1, (c.predictedTc ?? 0) / 300) + 0.5 * (c.uncertaintyEstimate ?? 0);
                      return (
                        <div key={c.id} className="flex items-center gap-3 p-2.5 bg-muted/30 rounded-md" data-testid={`dft-candidate-${i}`}>
                          <span className="text-xs font-mono text-muted-foreground w-4">#{i + 1}</span>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-sm font-mono font-medium truncate">{c.formula}</span>
                              {c.dataConfidence && (
                                <Badge variant={c.dataConfidence === "high" ? "default" : "secondary"} className="text-[10px]">
                                  {c.dataConfidence === "high" ? "DFT" : c.dataConfidence === "medium" ? "Model" : "Est."}
                                </Badge>
                              )}
                            </div>
                            <div className="flex items-center gap-3 mt-0.5 text-[10px] text-muted-foreground">
                              <span>Tc: <span className="font-mono font-bold text-foreground">{c.predictedTc ?? 0}K</span></span>
                              <span>Uncertainty: <span className="font-mono font-bold text-foreground">{((c.uncertaintyEstimate ?? 0) * 100).toFixed(0)}%</span></span>
                              <span>Acquisition: <span className="font-mono font-bold text-foreground">{(acqScore * 100).toFixed(0)}%</span></span>
                            </div>
                          </div>
                          <div className="w-16">
                            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${(c.uncertaintyEstimate ?? 0) > 0.6 ? "bg-red-500" : "bg-yellow-500"}`}
                                style={{ width: `${(c.uncertaintyEstimate ?? 0) * 100}%` }}
                              />
                            </div>
                          </div>
                          <Link href={`/candidate/${encodeURIComponent(c.formula)}`} data-testid={`link-dft-profile-${i}`}>
                            <ExternalLink className="h-3.5 w-3.5 text-primary cursor-pointer" />
                          </Link>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}

              {(dftStatus?.recentEnriched?.length ?? 0) > 0 && (
                <div className="space-y-2" data-testid="recent-enriched-list">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Recently DFT-Enriched</p>
                  {dftStatus!.recentEnriched.map((item, i) => (
                    <div key={i} className="flex items-center justify-between p-2 bg-muted/30 rounded-md text-xs" data-testid={`enriched-${i}`}>
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                        <span className="font-mono font-medium">{item.formula}</span>
                      </div>
                      <div className="flex items-center gap-3 text-muted-foreground">
                        <span>Tc: <span className="font-mono font-bold text-foreground">{item.predictedTc}K</span></span>
                        <Badge variant={item.confidence === "high" ? "default" : "secondary"} className="text-[10px]">
                          {item.confidence === "high" ? "DFT" : "Model"}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="structures" className="space-y-4">
          {structures.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {structures.map(s => <CrystalStructureCard key={s.id} structure={s} />)}
            </div>
          ) : (
            <Card>
              <CardContent className="pt-8 pb-8 text-center text-muted-foreground">
                <Layers className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Crystal structure predictions will appear as the pipeline processes candidates</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="failures" className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Negative results are crucial for understanding what does NOT work. Each rejected candidate narrows the search space.
          </p>
          {failedResults.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {failedResults.map(r => <FailedResultCard key={r.id} result={r} />)}
            </div>
          ) : (
            <Card>
              <CardContent className="pt-8 pb-8 text-center text-muted-foreground">
                <XCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No failed screenings yet. Failed candidates will be tracked here with failure reasons.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="synthesis" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Parameter Categories</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-synth-categories">
                  {synthesisData?.parameterSpace?.categories?.length ?? 0}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Total Variables</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-synth-variables">
                  {synthesisData?.parameterSpace?.totalVariables ?? 0}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Grid Points</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-synth-gridpoints">
                  {synthesisData?.parameterSpace?.totalGridPoints ?? 0}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Conditions Optimized</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-synth-optimized">
                  {synthesisData?.optimizerStats?.totalOptimized ?? 0}
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="card-synth-categories">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Layers className="h-5 w-5" />
                  Parameter Space Categories
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {(synthesisData?.parameterSpace?.categories ?? []).map((cat, i) => (
                    <div key={i} className="p-3 bg-muted/30 rounded-md" data-testid={`synth-category-${i}`}>
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
                {(synthesisData?.parameterSpace?.discreteVariables?.length ?? 0) > 0 && (
                  <div className="mt-4 pt-3 border-t">
                    <p className="text-xs font-medium text-muted-foreground mb-2">Discrete Variables</p>
                    <div className="space-y-1">
                      {synthesisData!.parameterSpace.discreteVariables.map((dv, i) => (
                        <div key={i} className="flex items-center justify-between text-xs" data-testid={`synth-discrete-${i}`}>
                          <span className="font-medium">{dv.name}</span>
                          <span className="text-muted-foreground">{dv.options.length} options</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="space-y-4">
              <Card data-testid="card-synth-feasibility">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Gauge className="h-5 w-5" />
                    Optimizer Performance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="p-2 bg-muted/30 rounded-md text-center">
                      <p className="text-xs text-muted-foreground">Avg Feasibility</p>
                      <p className="text-lg font-mono font-bold" data-testid="stat-avg-feasibility">
                        {((synthesisData?.optimizerStats?.avgFeasibility ?? 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2 bg-muted/30 rounded-md text-center">
                      <p className="text-xs text-muted-foreground">Total Optimized</p>
                      <p className="text-lg font-mono font-bold" data-testid="stat-total-optimized">
                        {synthesisData?.optimizerStats?.totalOptimized ?? 0}
                      </p>
                    </div>
                  </div>

                  {Object.keys(synthesisData?.optimizerStats?.methodBreakdown ?? {}).length > 0 && (
                    <div className="mb-3">
                      <p className="text-xs font-medium text-muted-foreground mb-1">Method Distribution</p>
                      <div className="space-y-1">
                        {Object.entries(synthesisData!.optimizerStats.methodBreakdown).map(([method, count]) => (
                          <div key={method} className="flex items-center justify-between text-xs" data-testid={`method-${method}`}>
                            <span>{method}</span>
                            <span className="font-mono">{count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {Object.keys(synthesisData?.optimizerStats?.complexityBreakdown ?? {}).length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground mb-1">Complexity Distribution</p>
                      <div className="flex gap-2">
                        {Object.entries(synthesisData!.optimizerStats.complexityBreakdown).map(([level, count]) => (
                          <Badge key={level} variant="outline" className="text-[10px]" data-testid={`complexity-${level}`}>
                            {level}: {count}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card data-testid="card-synth-top-conditions">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <FlaskConical className="h-5 w-5" />
                    Top Synthesis Conditions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {(synthesisData?.optimizerStats?.topConditions?.length ?? 0) > 0 ? (
                    <div className="space-y-2">
                      {synthesisData!.optimizerStats.topConditions.slice(0, 8).map((cond, i) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-muted/30 rounded-md text-xs" data-testid={`synth-cond-${i}`}>
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-medium">{cond.formula}</span>
                            <Badge variant="secondary" className="text-[10px]">{cond.method}</Badge>
                          </div>
                          <span className="font-mono">{(cond.feasibility * 100).toFixed(0)}% feasible</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      Synthesis conditions will appear as the engine optimizes candidates
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>

          {Object.keys(synthesisData?.optimizerStats?.categoryUsage ?? {}).length > 0 && (
            <Card data-testid="card-synth-category-usage">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Category Usage in Optimizations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
                  {Object.entries(synthesisData!.optimizerStats.categoryUsage)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cat, count]) => (
                      <div key={cat} className="p-2 bg-muted/30 rounded-md text-center" data-testid={`usage-${cat}`}>
                        <p className="text-[10px] text-muted-foreground truncate">{cat}</p>
                        <p className="text-sm font-mono font-bold">{count}</p>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Simulations Run</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-sim-total">
                  {simData?.simulator?.totalSimulations ?? 0}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Mutations Applied</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-sim-mutations">
                  {simData?.simulator?.totalMutations ?? 0}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Learning Records</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-learn-records">
                  {simData?.learning?.totalRecords ?? 0}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Unique Formulas</p>
                <p className="text-2xl font-mono font-bold" data-testid="stat-learn-unique">
                  {simData?.learning?.uniqueFormulas ?? 0}
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="card-sim-stats">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Synthesis Simulator
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div className="p-2 bg-muted/30 rounded-md text-center">
                    <p className="text-xs text-muted-foreground">Paths Optimized</p>
                    <p className="text-lg font-mono font-bold" data-testid="stat-paths-optimized">
                      {simData?.simulator?.totalPathsOptimized ?? 0}
                    </p>
                  </div>
                  <div className="p-2 bg-muted/30 rounded-md text-center">
                    <p className="text-xs text-muted-foreground">Best Tc Gain</p>
                    <p className="text-lg font-mono font-bold" data-testid="stat-best-tc-gain">
                      +{(simData?.simulator?.bestTcImprovement ?? 0).toFixed(1)}K
                    </p>
                  </div>
                </div>
                {simData?.simulator?.bestFormula && (
                  <div className="p-2 bg-muted/30 rounded-md text-xs mb-3">
                    <span className="text-muted-foreground">Best improved: </span>
                    <span className="font-mono font-medium" data-testid="stat-best-improved">{simData.simulator.bestFormula}</span>
                  </div>
                )}
                {Object.keys(simData?.simulator?.feasibilityBreakdown ?? {}).length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-1">Feasibility Classification</p>
                    <div className="flex gap-2">
                      {Object.entries(simData!.simulator.feasibilityBreakdown).map(([level, count]) => (
                        <Badge key={level} variant="outline" className="text-[10px]" data-testid={`feasibility-${level}`}>
                          {level}: {count}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                {Object.keys(simData?.simulator?.modeBreakdown ?? {}).length > 0 && (
                  <div className="mt-3">
                    <p className="text-xs font-medium text-muted-foreground mb-1">Optimization Modes</p>
                    <div className="flex gap-2">
                      {Object.entries(simData!.simulator.modeBreakdown)
                        .filter(([, c]) => c > 0)
                        .map(([mode, count]) => (
                          <Badge key={mode} variant="secondary" className="text-[10px]" data-testid={`mode-${mode}`}>
                            {mode}: {count}
                          </Badge>
                        ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-learn-patterns">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Star className="h-5 w-5" />
                  Learned Synthesis Patterns
                </CardTitle>
              </CardHeader>
              <CardContent>
                {(simData?.learning?.patterns?.length ?? 0) > 0 ? (
                  <div className="space-y-2">
                    {simData!.learning.patterns.slice(0, 6).map((pat, i) => (
                      <div key={i} className="p-2 bg-muted/30 rounded-md text-xs" data-testid={`pattern-${i}`}>
                        <p className="mb-1">{pat.description}</p>
                        <div className="flex gap-3 text-muted-foreground">
                          <span>Confidence: {(pat.confidence * 100).toFixed(0)}%</span>
                          <span>Samples: {pat.sampleCount}</span>
                          <span>Avg Tc+: {pat.avgTcImprovement.toFixed(1)}K</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    Patterns will emerge as the engine accumulates synthesis data
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {(simData?.learning?.parameterCorrelations?.length ?? 0) > 0 && (
            <Card data-testid="card-param-correlations">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Parameter-Tc Correlations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {simData!.learning.parameterCorrelations.map((pc, i) => (
                    <div key={i} className="p-2 bg-muted/30 rounded-md text-center" data-testid={`corr-${i}`}>
                      <p className="text-[10px] text-muted-foreground">{pc.parameter}</p>
                      <p className={`text-sm font-mono font-bold ${pc.correlation > 0 ? "text-green-500" : pc.correlation < -0.1 ? "text-red-500" : ""}`}>
                        {pc.correlation > 0 ? "+" : ""}{pc.correlation.toFixed(3)}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {Object.keys(simData?.learning?.classBreakdown ?? {}).length > 0 && (
            <Card data-testid="card-class-breakdown">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Beaker className="h-5 w-5" />
                  Synthesis by Material Class
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {Object.entries(simData!.learning.classBreakdown)
                    .sort(([, a], [, b]) => b.bestTc - a.bestTc)
                    .map(([cls, info]) => (
                      <div key={cls} className="p-2 bg-muted/30 rounded-md" data-testid={`class-${cls}`}>
                        <p className="text-xs font-medium truncate">{cls}</p>
                        <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                          <span>{info.count} records</span>
                          <span>Best: {info.bestTc.toFixed(0)}K</span>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="advanced-physics" className="space-y-4" data-testid="advanced-physics-content">
          <AdvancedPhysicsPanel />
        </TabsContent>

        <TabsContent value="next-gen-pipeline" className="space-y-4" data-testid="next-gen-pipeline-content">
          <NextGenPipelinePanel />
        </TabsContent>

        <TabsContent value="self-improving-lab" className="space-y-4" data-testid="self-improving-lab-content">
          <SelfImprovingLabPanel />
        </TabsContent>

        <TabsContent value="design-repr" className="space-y-4" data-testid="design-repr-content">
          <DesignRepresentationsPanel />
        </TabsContent>

        <TabsContent value="theory-discovery" className="space-y-4" data-testid="theory-discovery-content">
          <TheoryDiscoveryPanel />
        </TabsContent>

        <TabsContent value="causal-discovery" className="space-y-4" data-testid="causal-discovery-content">
          <CausalDiscoveryPanel />
        </TabsContent>

        <TabsContent value="dos-surrogate" className="space-y-4" data-testid="dos-surrogate-content">
          <DOSVisualizer />
        </TabsContent>
      </Tabs>
    </div>
  );
}

interface DesignReprStats {
  programs: {
    total: number;
    avgComplexity: number;
    avgInstructions: number;
    bestFormula: string;
    bestTc: number;
    instructionFrequency: Record<string, number>;
    generationDistribution: Record<number, number>;
  };
  graphs: {
    total: number;
    avgNodes: number;
    avgEdges: number;
    avgConnectivity: number;
    bestFormula: string;
    bestTc: number;
    componentFrequency: Record<string, number>;
    edgeTypeFrequency: Record<string, number>;
  };
  crossRepresentation: {
    programToGraphConversions: number;
    graphToProgramConversions: number;
    avgFeatureCorrelation: number;
  };
}

interface GeneratedProgram {
  program: {
    id: string;
    name: string;
    instructions: { type: string; params: Record<string, any>; order: number }[];
    outputFormula: string;
    outputPrototype: string;
    metadata: { complexity: number; expressiveness: number; generation: number; mutationHistory: string[] };
    featureVector: number[];
  };
  execution: {
    formula: string;
    prototype: string;
    elements: string[];
    stoichiometry: Record<string, number>;
    latticeType: string;
    symmetryGroup: string;
    hydrogenFraction: number;
    layerCount: number;
    channelDensity: number;
    strainApplied: number;
    dopingLevel: number;
    interfaceCount: number;
    complexity: number;
  };
}

interface GeneratedGraph {
  graph: {
    id: string;
    name: string;
    nodes: { id: string; type: string; element: string; properties: Record<string, any>; position: [number, number]; weight: number }[];
    edges: { source: string; target: string; type: string; strength: number; properties: Record<string, number> }[];
    outputFormula: string;
    metadata: { nodeCount: number; edgeCount: number; connectivity: number; avgDegree: number; clusteringCoeff: number; generation: number };
    embedding: number[];
  };
  analysis: {
    centralNodes: { id: string; element: string; centrality: number }[];
    bottleneckEdges: { source: string; target: string; betweenness: number }[];
    communities: { id: number; members: string[]; avgWeight: number }[];
    pathLengths: { avg: number; max: number; diameter: number };
    spectralGap: number;
    isConnected: boolean;
  };
}

function DesignRepresentationsPanel() {
  const [selectedStrategy, setSelectedStrategy] = useState("hydride-cage-optimizer");
  const [activeView, setActiveView] = useState<"program" | "graph" | "stats">("stats");
  const [generatedProgram, setGeneratedProgram] = useState<GeneratedProgram | null>(null);
  const [generatedGraph, setGeneratedGraph] = useState<GeneratedGraph | null>(null);

  const statsQuery = useQuery<DesignReprStats>({
    queryKey: ["/api/design-representations/stats"],
    refetchInterval: 10000,
  });

  const generateProgramMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/design-representations/program/generate", {
        strategyType: selectedStrategy,
        elementPool: ["La", "Y", "H", "Nb", "Cu", "O", "Ba", "Sr"],
      });
      return res.json();
    },
    onSuccess: (data: GeneratedProgram) => {
      setGeneratedProgram(data);
      setActiveView("program");
      queryClient.invalidateQueries({ queryKey: ["/api/design-representations/stats"] });
    },
  });

  const mutateProgramMutation = useMutation({
    mutationFn: async () => {
      if (!generatedProgram) return;
      const res = await apiRequest("POST", "/api/design-representations/program/mutate", {
        program: generatedProgram.program,
        elementPool: ["La", "Y", "H", "Nb", "Cu", "O", "Ba", "Sr", "Ti", "Zr"],
      });
      return res.json();
    },
    onSuccess: (data: GeneratedProgram) => {
      if (data) setGeneratedProgram(data);
    },
  });

  const generateGraphMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/design-representations/graph/generate", {
        strategyType: selectedStrategy,
        elementPool: ["La", "Y", "H", "Nb", "Cu", "O", "Ba", "Sr"],
      });
      return res.json();
    },
    onSuccess: (data: GeneratedGraph) => {
      setGeneratedGraph(data);
      setActiveView("graph");
      queryClient.invalidateQueries({ queryKey: ["/api/design-representations/stats"] });
    },
  });

  const mutateGraphMutation = useMutation({
    mutationFn: async () => {
      if (!generatedGraph) return;
      const res = await apiRequest("POST", "/api/design-representations/graph/mutate", {
        graph: generatedGraph.graph,
        elementPool: ["La", "Y", "H", "Nb", "Cu", "O", "Ba", "Sr", "Bi", "Se"],
      });
      return res.json();
    },
    onSuccess: (data: GeneratedGraph) => {
      if (data) setGeneratedGraph(data);
    },
  });

  const convertProgramToGraphMutation = useMutation({
    mutationFn: async () => {
      if (!generatedProgram) return;
      const res = await apiRequest("POST", "/api/design-representations/convert/program-to-graph", {
        program: generatedProgram.program,
      });
      return res.json();
    },
    onSuccess: (data: GeneratedGraph) => {
      if (data) {
        setGeneratedGraph(data);
        setActiveView("graph");
      }
    },
  });

  const convertGraphToProgramMutation = useMutation({
    mutationFn: async () => {
      if (!generatedGraph) return;
      const res = await apiRequest("POST", "/api/design-representations/convert/graph-to-program", {
        graph: generatedGraph.graph,
      });
      return res.json();
    },
    onSuccess: (data: GeneratedProgram) => {
      if (data) {
        setGeneratedProgram(data);
        setActiveView("program");
      }
    },
  });

  const stats = statsQuery.data;
  const strategies = [
    "hydride-cage-optimizer", "layered-intercalation", "high-entropy-alloy",
    "light-element-phonon", "topological-edge", "pressure-stabilized",
    "electron-phonon-resonance", "charge-transfer-layer",
  ];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code2 className="h-5 w-5 text-purple-500" />
            Design Representations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Dual representation system: Programmatic (procedural structure generation) and
            Graph-based (component-connection architecture). Both representations are interconvertible
            and integrated into the design lab iteration cycle.
          </p>

          <div className="flex flex-wrap gap-2 mb-4">
            <Button
              variant={activeView === "stats" ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveView("stats")}
              data-testid="btn-view-stats"
            >
              <Activity className="h-3.5 w-3.5 mr-1" /> Stats
            </Button>
            <Button
              variant={activeView === "program" ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveView("program")}
              data-testid="btn-view-program"
            >
              <Code2 className="h-3.5 w-3.5 mr-1" /> Program View
            </Button>
            <Button
              variant={activeView === "graph" ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveView("graph")}
              data-testid="btn-view-graph"
            >
              <Network className="h-3.5 w-3.5 mr-1" /> Graph View
            </Button>
          </div>

          <div className="flex flex-wrap gap-2 mb-4">
            <select
              className="text-sm border rounded px-2 py-1 bg-background"
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              data-testid="select-strategy"
            >
              {strategies.map(s => (
                <option key={s} value={s}>{s.split("-").map(w => w[0].toUpperCase() + w.slice(1)).join(" ")}</option>
              ))}
            </select>
            <Button
              size="sm"
              onClick={() => generateProgramMutation.mutate()}
              disabled={generateProgramMutation.isPending}
              data-testid="btn-generate-program"
            >
              {generateProgramMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <Code2 className="h-3.5 w-3.5 mr-1" />}
              Generate Program
            </Button>
            <Button
              size="sm"
              onClick={() => generateGraphMutation.mutate()}
              disabled={generateGraphMutation.isPending}
              data-testid="btn-generate-graph"
            >
              {generateGraphMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <Network className="h-3.5 w-3.5 mr-1" />}
              Generate Graph
            </Button>
          </div>
        </CardContent>
      </Card>

      {activeView === "stats" && stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Code2 className="h-4 w-4 text-blue-500" /> Procedural Programs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Total Programs</span><span className="font-medium" data-testid="text-program-total">{stats.programs.total}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Avg Complexity</span><span className="font-medium">{stats.programs.avgComplexity.toFixed(2)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Avg Instructions</span><span className="font-medium">{stats.programs.avgInstructions.toFixed(1)}</span></div>
                {stats.programs.bestFormula && (
                  <div className="flex justify-between"><span className="text-muted-foreground">Best Formula</span><span className="font-medium" data-testid="text-program-best">{stats.programs.bestFormula} ({stats.programs.bestTc}K)</span></div>
                )}
                {Object.keys(stats.programs.instructionFrequency).length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs text-muted-foreground mb-1">Instruction Usage</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(stats.programs.instructionFrequency)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 6)
                        .map(([type, count]) => (
                          <Badge key={type} variant="secondary" className="text-xs">
                            {type.replace(/_/g, " ")} ({count})
                          </Badge>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Network className="h-4 w-4 text-green-500" /> Component Graphs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Total Graphs</span><span className="font-medium" data-testid="text-graph-total">{stats.graphs.total}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Avg Nodes</span><span className="font-medium">{stats.graphs.avgNodes.toFixed(1)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Avg Edges</span><span className="font-medium">{stats.graphs.avgEdges.toFixed(1)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Avg Connectivity</span><span className="font-medium">{stats.graphs.avgConnectivity.toFixed(3)}</span></div>
                {stats.graphs.bestFormula && (
                  <div className="flex justify-between"><span className="text-muted-foreground">Best Formula</span><span className="font-medium" data-testid="text-graph-best">{stats.graphs.bestFormula} ({stats.graphs.bestTc}K)</span></div>
                )}
                {Object.keys(stats.graphs.componentFrequency).length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs text-muted-foreground mb-1">Component Types</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(stats.graphs.componentFrequency)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 6)
                        .map(([type, count]) => (
                          <Badge key={type} variant="secondary" className="text-xs">
                            {type.replace(/_/g, " ")} ({count})
                          </Badge>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <ArrowLeftRight className="h-4 w-4 text-amber-500" /> Cross-Representation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Program to Graph</span><span className="font-medium">{stats.crossRepresentation.programToGraphConversions}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Graph to Program</span><span className="font-medium">{stats.crossRepresentation.graphToProgramConversions}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Feature Correlation</span><span className="font-medium">{stats.crossRepresentation.avgFeatureCorrelation.toFixed(2)}</span></div>
                {Object.keys(stats.graphs.edgeTypeFrequency).length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs text-muted-foreground mb-1">Edge Types</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(stats.graphs.edgeTypeFrequency)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 5)
                        .map(([type, count]) => (
                          <Badge key={type} variant="secondary" className="text-xs">
                            {type.replace(/_/g, " ")} ({count})
                          </Badge>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "program" && generatedProgram && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Code2 className="h-4 w-4 text-blue-500" /> Design Program
                <Badge variant="outline" className="ml-auto">{generatedProgram.program.name}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-muted rounded-md p-3 font-mono text-xs space-y-1 mb-3 max-h-72 overflow-y-auto" data-testid="program-code-view">
                <div className="text-muted-foreground">{"// Procedural structure generator"}</div>
                <div className="text-blue-500">function generate_structure() {"{"}</div>
                {generatedProgram.program.instructions
                  .sort((a: { order: number }, b: { order: number }) => a.order - b.order)
                  .map((inst: { type: string; params: Record<string, any>; order: number }, i: number) => (
                  <div key={i} className="pl-4">
                    <span className="text-green-600">{inst.type}</span>
                    <span className="text-muted-foreground">(</span>
                    <span className="text-amber-600">
                      {Object.entries(inst.params)
                        .filter(([k]) => k !== "order")
                        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                        .join(", ")}
                    </span>
                    <span className="text-muted-foreground">)</span>
                  </div>
                ))}
                <div className="text-blue-500">{"}"}</div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={() => mutateProgramMutation.mutate()} disabled={mutateProgramMutation.isPending} data-testid="btn-mutate-program">
                  {mutateProgramMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <GitBranch className="h-3.5 w-3.5 mr-1" />}
                  Mutate
                </Button>
                <Button size="sm" variant="outline" onClick={() => convertProgramToGraphMutation.mutate()} disabled={convertProgramToGraphMutation.isPending} data-testid="btn-convert-to-graph">
                  {convertProgramToGraphMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <ArrowLeftRight className="h-3.5 w-3.5 mr-1" />}
                  Convert to Graph
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Beaker className="h-4 w-4 text-purple-500" /> Execution Result
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Formula</span><span className="font-bold text-lg" data-testid="text-exec-formula">{generatedProgram.execution.formula}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Prototype</span><span className="font-medium">{generatedProgram.execution.prototype}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Lattice</span><span className="font-medium">{generatedProgram.execution.latticeType}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Elements</span><span className="font-medium">{generatedProgram.execution.elements.join(", ")}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Complexity</span><span className="font-medium">{generatedProgram.execution.complexity.toFixed(2)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">H Fraction</span><span className="font-medium">{(generatedProgram.execution.hydrogenFraction * 100).toFixed(1)}%</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Layers</span><span className="font-medium">{generatedProgram.execution.layerCount}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Interfaces</span><span className="font-medium">{generatedProgram.execution.interfaceCount}</span></div>
                {generatedProgram.execution.strainApplied > 0 && (
                  <div className="flex justify-between"><span className="text-muted-foreground">Strain</span><span className="font-medium">{(generatedProgram.execution.strainApplied * 100).toFixed(1)}%</span></div>
                )}
                {generatedProgram.execution.dopingLevel > 0 && (
                  <div className="flex justify-between"><span className="text-muted-foreground">Doping</span><span className="font-medium">{(generatedProgram.execution.dopingLevel * 100).toFixed(1)}%</span></div>
                )}
                <div className="mt-2">
                  <div className="text-xs text-muted-foreground mb-1">Stoichiometry</div>
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(generatedProgram.execution.stoichiometry).map(([el, count]) => (
                      <Badge key={el} variant="outline" className="text-xs">{el}: {count as number}</Badge>
                    ))}
                  </div>
                </div>
                <div className="mt-2">
                  <div className="text-xs text-muted-foreground mb-1">Feature Vector ({generatedProgram.program.featureVector.length}D)</div>
                  <div className="bg-muted rounded p-1.5 text-xs font-mono truncate">
                    [{generatedProgram.program.featureVector.map((v: number) => v.toFixed(3)).join(", ")}]
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "graph" && generatedGraph && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Network className="h-4 w-4 text-green-500" /> Component Graph
                <Badge variant="outline" className="ml-auto">{generatedGraph.graph.name}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative bg-muted rounded-md mb-3 overflow-hidden" style={{ height: 260 }} data-testid="graph-visual">
                <svg width="100%" height="100%" viewBox="0 0 400 260">
                  {generatedGraph.graph.edges.map((edge, i) => {
                    const src = generatedGraph.graph.nodes.find(n => n.id === edge.source);
                    const tgt = generatedGraph.graph.nodes.find(n => n.id === edge.target);
                    if (!src || !tgt) return null;
                    const x1 = src.position[0] * 360 + 20;
                    const y1 = src.position[1] * 220 + 20;
                    const x2 = tgt.position[0] * 360 + 20;
                    const y2 = tgt.position[1] * 220 + 20;
                    const colors: Record<string, string> = {
                      bonding: "#3b82f6", electron_transfer: "#ef4444", phonon_coupling: "#22c55e",
                      charge_transfer: "#f59e0b", structural: "#6b7280", proximity: "#a855f7",
                      epitaxial: "#06b6d4", hybridization: "#ec4899",
                    };
                    return (
                      <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke={colors[edge.type] || "#6b7280"}
                        strokeWidth={Math.max(1, edge.strength * 3)}
                        strokeOpacity={0.6} />
                    );
                  })}
                  {generatedGraph.graph.nodes.map((node, i) => {
                    const cx = node.position[0] * 360 + 20;
                    const cy = node.position[1] * 220 + 20;
                    const r = 14 + node.weight * 8;
                    const colors: Record<string, string> = {
                      electron_source: "#ef4444", phonon_mediator: "#22c55e", charge_reservoir: "#f59e0b",
                      structural_backbone: "#3b82f6", hydrogen_cage: "#06b6d4", intercalation_host: "#8b5cf6",
                      dopant_site: "#ec4899", interface_layer: "#14b8a6", strain_buffer: "#6b7280",
                      topological_surface: "#a855f7", pairing_channel: "#f97316", dos_enhancer: "#64748b",
                    };
                    return (
                      <g key={i}>
                        <circle cx={cx} cy={cy} r={r} fill={colors[node.type] || "#6b7280"} fillOpacity={0.85} stroke="white" strokeWidth={1.5} />
                        <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central" fill="white" fontSize={10} fontWeight="bold">
                          {node.element}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="flex flex-wrap gap-1 mb-3">
                {generatedGraph.graph.nodes.map((node, i) => (
                  <Badge key={i} variant="outline" className="text-xs">
                    {node.element} ({node.type.replace(/_/g, " ")})
                  </Badge>
                ))}
              </div>

              <div className="text-xs text-muted-foreground mb-2">Edges</div>
              <div className="space-y-1 max-h-28 overflow-y-auto mb-3">
                {generatedGraph.graph.edges.map((edge, i) => {
                  const src = generatedGraph.graph.nodes.find(n => n.id === edge.source);
                  const tgt = generatedGraph.graph.nodes.find(n => n.id === edge.target);
                  return (
                    <div key={i} className="flex items-center gap-1.5 text-xs">
                      <span className="font-medium">{src?.element || "?"}</span>
                      <span className="text-muted-foreground">--[{edge.type.replace(/_/g, " ")} {(edge.strength * 100).toFixed(0)}%]--</span>
                      <span className="font-medium">{tgt?.element || "?"}</span>
                    </div>
                  );
                })}
              </div>

              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={() => mutateGraphMutation.mutate()} disabled={mutateGraphMutation.isPending} data-testid="btn-mutate-graph">
                  {mutateGraphMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <GitBranch className="h-3.5 w-3.5 mr-1" />}
                  Mutate
                </Button>
                <Button size="sm" variant="outline" onClick={() => convertGraphToProgramMutation.mutate()} disabled={convertGraphToProgramMutation.isPending} data-testid="btn-convert-to-program">
                  {convertGraphToProgramMutation.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <ArrowLeftRight className="h-3.5 w-3.5 mr-1" />}
                  Convert to Program
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Brain className="h-4 w-4 text-amber-500" /> Graph Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Formula</span><span className="font-bold text-lg" data-testid="text-graph-formula">{generatedGraph.graph.outputFormula}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Nodes</span><span className="font-medium">{generatedGraph.graph.metadata.nodeCount}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Edges</span><span className="font-medium">{generatedGraph.graph.metadata.edgeCount}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Connectivity</span><span className="font-medium">{generatedGraph.graph.metadata.connectivity.toFixed(3)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Avg Degree</span><span className="font-medium">{generatedGraph.graph.metadata.avgDegree.toFixed(2)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Clustering</span><span className="font-medium">{generatedGraph.graph.metadata.clusteringCoeff.toFixed(3)}</span></div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Connected</span>
                  <span className="font-medium">{generatedGraph.analysis.isConnected ? "Yes" : "No"}</span>
                </div>
                <div className="flex justify-between"><span className="text-muted-foreground">Spectral Gap</span><span className="font-medium">{generatedGraph.analysis.spectralGap.toFixed(3)}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Path Avg/Max</span><span className="font-medium">{generatedGraph.analysis.pathLengths.avg.toFixed(2)} / {generatedGraph.analysis.pathLengths.max}</span></div>

                <div className="mt-3">
                  <div className="text-xs text-muted-foreground mb-1">Central Nodes</div>
                  <div className="space-y-0.5">
                    {generatedGraph.analysis.centralNodes.map((cn, i) => (
                      <div key={i} className="flex justify-between text-xs">
                        <span>{cn.element} ({cn.id})</span>
                        <span className="font-medium">centrality: {cn.centrality.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-2">
                  <div className="text-xs text-muted-foreground mb-1">Communities ({generatedGraph.analysis.communities.length})</div>
                  <div className="flex flex-wrap gap-1">
                    {generatedGraph.analysis.communities.map((c) => (
                      <Badge key={c.id} variant="secondary" className="text-xs">
                        Cluster {c.id}: {c.members.length} nodes (w={c.avgWeight.toFixed(2)})
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="mt-2">
                  <div className="text-xs text-muted-foreground mb-1">Graph Embedding ({generatedGraph.graph.embedding.length}D)</div>
                  <div className="bg-muted rounded p-1.5 text-xs font-mono truncate">
                    [{generatedGraph.graph.embedding.map((v: number) => v.toFixed(3)).join(", ")}]
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "program" && !generatedProgram && (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            Click "Generate Program" to create a procedural design representation.
          </CardContent>
        </Card>
      )}

      {activeView === "graph" && !generatedGraph && (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            Click "Generate Graph" to create a component graph representation.
          </CardContent>
        </Card>
      )}

      {statsQuery.isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[0, 1, 2].map(i => (
            <Card key={i}>
              <CardContent className="pt-6">
                <Skeleton className="h-4 w-32 mb-2" />
                <Skeleton className="h-4 w-full mb-1" />
                <Skeleton className="h-4 w-3/4 mb-1" />
                <Skeleton className="h-4 w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

interface TheoryData {
  id: string;
  equation: string;
  simplified: string;
  target: string;
  theoryScore: number;
  accuracy: number;
  mae: number;
  complexity: number;
  simplicity: number;
  generalization: number;
  physicsCompliance: number;
  novelty: number;
  dimensionallyValid: boolean;
  variables: string[];
  applicableFamilies: string[];
  crossScaleValidation: { family: string; sampleCount: number; r2: number; mae: number }[];
  featureImportance: { variable: string; importance: number }[];
  discoveredAt: number;
}

interface DiscoveryStats {
  totalRuns: number;
  totalEquationsEvaluated: number;
  theoriesDiscovered: number;
  featureLibrarySize: number;
  averageTheoryScore: number;
  bestTheoryScore: number;
  physicsVariablesUsed: string[];
  familyCoverage: Record<string, number>;
  topTheories: {
    id: string;
    equation: string;
    simplified: string;
    theoryScore: number;
    r2: number;
    complexity: number;
    generalization: number;
    physicsCompliance: number;
    novelty: number;
    dimensionallyValid: boolean;
    variables: string[];
    applicableFamilies: string[];
  }[];
  unitRegistry: { variable: string; unit: string }[];
}

interface FeatureLibraryItem {
  name: string;
  expression: string;
  variables: string[];
  category: string;
  physicsInspired: boolean;
}

function TheoryDiscoveryPanel() {
  const [activeView, setActiveView] = useState<"overview" | "theories" | "features" | "units" | "feedback">("overview");
  const [familyFilter, setFamilyFilter] = useState<string>("all");
  const [mechanismFilter, setMechanismFilter] = useState<string>("all");

  const statsQuery = useQuery<DiscoveryStats>({
    queryKey: ["/api/symbolic-discovery/stats"],
  });

  const theoriesQuery = useQuery<TheoryData[]>({
    queryKey: ["/api/symbolic-discovery/theories"],
  });

  const featureLibraryQuery = useQuery<FeatureLibraryItem[]>({
    queryKey: ["/api/symbolic-discovery/feature-library"],
  });

  const feedbackQuery = useQuery<{
    biasedVariables: { variable: string; direction: string; strength: number }[];
    suggestedCompositions: string[];
    theoreticalInsight: string;
  }>({
    queryKey: ["/api/symbolic-discovery/feedback"],
  });

  const runDiscoveryMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/symbolic-discovery/run", {});
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/symbolic-discovery/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/symbolic-discovery/theories"] });
      queryClient.invalidateQueries({ queryKey: ["/api/symbolic-discovery/feedback"] });
    },
  });

  const stats = statsQuery.data;
  const theories = theoriesQuery.data ?? [];
  const features = featureLibraryQuery.data ?? [];
  const feedback = feedbackQuery.data;

  const filteredTheories = theories.filter(t => {
    if (familyFilter !== "all" && !t.applicableFamilies.includes(familyFilter)) return false;
    return true;
  });

  const allFamilies = [...new Set(theories.flatMap(t => t.applicableFamilies))];

  return (
    <div className="space-y-4" data-testid="theory-discovery-panel">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-theory-discovery-title">
            <Sigma className="h-5 w-5 text-violet-500" />
            Symbolic Physics Discovery Layer
          </h3>
          <p className="text-sm text-muted-foreground">
            Multi-objective symbolic equation search with physics constraints, dimensional analysis, and cross-scale validation
          </p>
        </div>
        <Button
          onClick={() => runDiscoveryMutation.mutate()}
          disabled={runDiscoveryMutation.isPending}
          data-testid="button-run-discovery"
        >
          {runDiscoveryMutation.isPending ? (
            <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Running...</>
          ) : (
            <><Search className="h-4 w-4 mr-2" />Run Discovery</>
          )}
        </Button>
      </div>

      <div className="flex gap-2 flex-wrap">
        {(["overview", "theories", "features", "units", "feedback"] as const).map(view => (
          <Button
            key={view}
            variant={activeView === view ? "default" : "outline"}
            size="sm"
            onClick={() => setActiveView(view)}
            data-testid={`button-view-${view}`}
          >
            {view === "overview" && <><Activity className="h-3 w-3 mr-1" />Overview</>}
            {view === "theories" && <><BookOpen className="h-3 w-3 mr-1" />Theories</>}
            {view === "features" && <><Layers className="h-3 w-3 mr-1" />Feature Library</>}
            {view === "units" && <><ShieldCheck className="h-3 w-3 mr-1" />Unit Registry</>}
            {view === "feedback" && <><Lightbulb className="h-3 w-3 mr-1" />Feedback Loop</>}
          </Button>
        ))}
      </div>

      {activeView === "overview" && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold" data-testid="text-total-runs">{stats?.totalRuns ?? 0}</div>
                <div className="text-xs text-muted-foreground">Discovery Runs</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold" data-testid="text-equations-evaluated">{(stats?.totalEquationsEvaluated ?? 0).toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">Equations Evaluated</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold" data-testid="text-theories-discovered">{stats?.theoriesDiscovered ?? 0}</div>
                <div className="text-xs text-muted-foreground">Theories Discovered</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold" data-testid="text-best-score">{stats?.bestTheoryScore ?? 0}</div>
                <div className="text-xs text-muted-foreground">Best Theory Score</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-emerald-500" />
                  Top Discovered Equations
                </CardTitle>
              </CardHeader>
              <CardContent>
                {stats?.topTheories && stats.topTheories.length > 0 ? (
                  <div className="space-y-3">
                    {stats.topTheories.slice(0, 5).map((t, i) => (
                      <div key={t.id} className="border rounded-lg p-3 space-y-1" data-testid={`card-top-theory-${i}`}>
                        <div className="flex items-center justify-between">
                          <code className="text-xs bg-muted px-2 py-1 rounded font-mono max-w-[70%] overflow-hidden text-ellipsis" data-testid={`text-theory-equation-${i}`}>
                            {t.simplified}
                          </code>
                          <Badge variant={t.theoryScore > 0.5 ? "default" : "secondary"} data-testid={`badge-theory-score-${i}`}>
                            {t.theoryScore}
                          </Badge>
                        </div>
                        <div className="flex gap-2 text-xs text-muted-foreground">
                          <span>R2: {t.r2}</span>
                          <span>Gen: {t.generalization}</span>
                          <span>Complexity: {t.complexity}</span>
                          {t.dimensionallyValid && <Badge variant="outline" className="text-[10px] px-1">Dim Valid</Badge>}
                        </div>
                        <div className="flex gap-1 flex-wrap">
                          {t.applicableFamilies.map(f => (
                            <Badge key={f} variant="outline" className="text-[10px] px-1">{f}</Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground" data-testid="text-no-theories">
                    No theories discovered yet. Click "Run Discovery" to start symbolic equation search.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Target className="h-4 w-4 text-blue-500" />
                  Material Family Coverage
                </CardTitle>
              </CardHeader>
              <CardContent>
                {stats?.familyCoverage && Object.keys(stats.familyCoverage).length > 0 ? (
                  <div className="space-y-2">
                    {Object.entries(stats.familyCoverage).sort((a, b) => b[1] - a[1]).map(([family, count]) => (
                      <div key={family} className="flex items-center justify-between" data-testid={`row-family-${family}`}>
                        <span className="text-sm">{family}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-24 bg-muted rounded-full h-2">
                            <div
                              className="bg-violet-500 rounded-full h-2"
                              style={{ width: `${Math.min(100, (count / Math.max(1, stats.theoriesDiscovered)) * 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground w-8 text-right">{count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No family coverage data yet.</p>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Sigma className="h-4 w-4 text-violet-500" />
                Discovery Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-7 gap-1 text-center text-xs">
                {[
                  { label: "Simulation Data", icon: "data", desc: "Multi-scale features" },
                  { label: "Feature Library", icon: "lib", desc: `${stats?.featureLibrarySize ?? 0} terms` },
                  { label: "Equation Search", icon: "search", desc: "Evolutionary GP" },
                  { label: "Physics Filter", icon: "filter", desc: "Constraint validation" },
                  { label: "Cross-Scale", icon: "scale", desc: "Family generalization" },
                  { label: "Theory Ranking", icon: "rank", desc: "Multi-objective score" },
                  { label: "Integration", icon: "integrate", desc: "Feedback loop" },
                ].map((stage, i) => (
                  <div key={i} className="flex flex-col items-center gap-1">
                    <div className="w-10 h-10 rounded-full bg-violet-500/10 border border-violet-500/30 flex items-center justify-center text-violet-500 font-bold">
                      {i + 1}
                    </div>
                    <span className="font-medium">{stage.label}</span>
                    <span className="text-muted-foreground">{stage.desc}</span>
                    {i < 6 && <ArrowDown className="h-3 w-3 text-muted-foreground mt-1" />}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {activeView === "theories" && (
        <>
          <div className="flex gap-2 items-center">
            <span className="text-sm text-muted-foreground">Filter by family:</span>
            <select
              className="text-sm border rounded px-2 py-1 bg-background"
              value={familyFilter}
              onChange={e => setFamilyFilter(e.target.value)}
              data-testid="select-family-filter"
            >
              <option value="all">All Families</option>
              {allFamilies.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          {filteredTheories.length > 0 ? (
            <div className="space-y-3">
              {filteredTheories.map((theory, i) => (
                <Card key={theory.id} data-testid={`card-theory-${i}`}>
                  <CardContent className="pt-4 space-y-3">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1 flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Badge variant="default" data-testid={`badge-theory-rank-${i}`}>#{i + 1}</Badge>
                          <span className="text-xs text-muted-foreground">Score: {theory.theoryScore}</span>
                          {theory.dimensionallyValid && (
                            <Badge variant="outline" className="text-green-600 border-green-300">
                              <ShieldCheck className="h-3 w-3 mr-1" />Dim Valid
                            </Badge>
                          )}
                        </div>
                        <div className="mt-1">
                          <code className="text-sm bg-muted px-2 py-1 rounded font-mono block overflow-x-auto" data-testid={`text-theory-full-eq-${i}`}>
                            Tc = {theory.simplified}
                          </code>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                      <div>
                        <span className="text-muted-foreground">Accuracy (R2)</span>
                        <div className="font-medium" data-testid={`text-theory-accuracy-${i}`}>{theory.accuracy}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Simplicity</span>
                        <div className="font-medium">{theory.simplicity}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Generalization</span>
                        <div className="font-medium">{theory.generalization}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Physics</span>
                        <div className="font-medium">{theory.physicsCompliance}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Novelty</span>
                        <div className="font-medium">{theory.novelty}</div>
                      </div>
                    </div>

                    <div className="text-xs text-muted-foreground">
                      Score = 0.35*accuracy + 0.20*simplicity + 0.20*generalization + 0.15*physics + 0.10*novelty
                    </div>

                    {theory.crossScaleValidation.length > 0 && (
                      <div>
                        <span className="text-xs font-medium">Cross-Scale Validation:</span>
                        <div className="flex gap-2 flex-wrap mt-1">
                          {theory.crossScaleValidation.map(csv => (
                            <Badge
                              key={csv.family}
                              variant={csv.r2 > 0.3 ? "default" : "secondary"}
                              className="text-[10px]"
                              data-testid={`badge-csv-${csv.family}-${i}`}
                            >
                              {csv.family}: R2={csv.r2.toFixed(2)} (n={csv.sampleCount})
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {theory.featureImportance.length > 0 && (
                      <div>
                        <span className="text-xs font-medium">Feature Importance:</span>
                        <div className="flex gap-1 flex-wrap mt-1">
                          {theory.featureImportance.slice(0, 5).map(fi => (
                            <Badge key={fi.variable} variant="outline" className="text-[10px]">
                              {fi.variable}: {fi.importance.toFixed(3)}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex gap-1 flex-wrap">
                      {theory.variables.map(v => (
                        <Badge key={v} variant="outline" className="text-[10px] bg-violet-50 dark:bg-violet-950">{v}</Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground" data-testid="text-no-filtered-theories">
                {theories.length === 0
                  ? "No theories discovered yet. Run a discovery cycle first."
                  : "No theories match the selected filter."}
              </CardContent>
            </Card>
          )}
        </>
      )}

      {activeView === "features" && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Layers className="h-4 w-4 text-amber-500" />
                Symbolic Feature Library ({features.length} terms)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {["basic", "quadratic", "interaction", "ratio", "transform", "physics"].map(category => {
                  const catFeatures = features.filter(f => f.category === category);
                  if (catFeatures.length === 0) return null;
                  return (
                    <div key={category}>
                      <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">{category} ({catFeatures.length})</h4>
                      <div className="flex gap-1 flex-wrap">
                        {catFeatures.map(f => (
                          <Badge
                            key={f.name}
                            variant={f.physicsInspired ? "default" : "outline"}
                            className="text-[10px] font-mono"
                            data-testid={`badge-feature-${f.name}`}
                          >
                            {f.expression}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "units" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-500" />
              Dimensional Analysis Unit Registry
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {stats?.unitRegistry?.map(entry => (
                <div key={entry.variable} className="flex items-center justify-between border rounded px-3 py-2 text-sm" data-testid={`row-unit-${entry.variable}`}>
                  <code className="font-mono text-xs">{entry.variable}</code>
                  <Badge variant="outline" className="text-[10px] font-mono">{entry.unit}</Badge>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-3">
              Equations with mismatched left/right hand side units are automatically rejected during discovery.
              Only dimensionally consistent theories pass the physics constraint filter.
            </p>
          </CardContent>
        </Card>
      )}

      {activeView === "feedback" && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-yellow-500" />
                Discovery Feedback Loop
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {feedback ? (
                <>
                  <div>
                    <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Theoretical Insight</h4>
                    <p className="text-sm" data-testid="text-theoretical-insight">{feedback.theoreticalInsight}</p>
                  </div>

                  {feedback.biasedVariables.length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Generator Bias from Discovered Theories</h4>
                      <div className="space-y-2">
                        {feedback.biasedVariables.map(bv => (
                          <div key={bv.variable} className="flex items-center gap-3" data-testid={`row-bias-${bv.variable}`}>
                            <code className="text-xs font-mono w-40">{bv.variable}</code>
                            <Badge variant={bv.direction === "increase" ? "default" : "secondary"}>
                              {bv.direction}
                            </Badge>
                            <div className="flex-1 bg-muted rounded-full h-2">
                              <div
                                className={`rounded-full h-2 ${bv.direction === "increase" ? "bg-emerald-500" : "bg-orange-500"}`}
                                style={{ width: `${bv.strength * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground">{(bv.strength * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {feedback.suggestedCompositions.length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Suggested Material Directions</h4>
                      <div className="space-y-1">
                        {feedback.suggestedCompositions.map((s, i) => (
                          <div key={i} className="text-sm flex items-start gap-2" data-testid={`text-suggestion-${i}`}>
                            <FlaskRound className="h-4 w-4 mt-0.5 text-violet-500 flex-shrink-0" />
                            {s}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted-foreground" data-testid="text-no-feedback">
                  Run a discovery cycle to generate feedback for the candidate generator.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-blue-500" />
                Theory-Design Integration Cycle
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-2 text-center text-xs">
                {[
                  { label: "Discovered Theory", desc: "Symbolic equation for Tc" },
                  { label: "Feature Analysis", desc: "Key variables identified" },
                  { label: "Generator Bias", desc: "Candidate search directed" },
                  { label: "New Data", desc: "Validates / refines theory" },
                ].map((step, i) => (
                  <div key={i} className="flex flex-col items-center gap-1 p-2 border rounded">
                    <div className="w-8 h-8 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-500 font-bold text-xs">
                      {i + 1}
                    </div>
                    <span className="font-medium">{step.label}</span>
                    <span className="text-muted-foreground">{step.desc}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {statsQuery.isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[0, 1].map(i => (
            <Card key={i}>
              <CardContent className="pt-6">
                <Skeleton className="h-4 w-32 mb-2" />
                <Skeleton className="h-4 w-full mb-1" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function CausalDiscoveryPanel() {
  const [causalView, setCausalView] = useState<"overview" | "graph" | "hypotheses" | "intervention" | "ontology" | "rules" | "pressure">("overview");
  const [interventionFormula, setInterventionFormula] = useState("LaH10");
  const [interventionVar, setInterventionVar] = useState("pressure");
  const [interventionValue, setInterventionValue] = useState("100");
  const [counterfactualVar, setCounterfactualVar] = useState("phonon_freq");
  const [counterfactualPct, setCounterfactualPct] = useState("20");

  const statsQuery = useQuery<any>({ queryKey: ["/api/causal-discovery/stats"] });
  const variablesQuery = useQuery<any[]>({ queryKey: ["/api/causal-discovery/variables"] });
  const ontologyQuery = useQuery<any[]>({ queryKey: ["/api/causal-discovery/ontology"] });
  const graphQuery = useQuery<any>({ queryKey: ["/api/causal-discovery/graph"] });
  const hypothesesQuery = useQuery<any[]>({ queryKey: ["/api/causal-discovery/hypotheses"] });
  const rulesQuery = useQuery<any[]>({ queryKey: ["/api/causal-discovery/rules"] });

  const runMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/causal-discovery/run", { datasetSize: 60 }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/graph"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/hypotheses"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/rules"] });
    },
  });

  const interventionMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/causal-discovery/intervene", {
      formula: interventionFormula,
      variable: interventionVar,
      newValue: parseFloat(interventionValue),
    }),
  });

  const counterfactualMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/causal-discovery/counterfactual", {
      formula: interventionFormula,
      variable: counterfactualVar,
      modificationPercent: parseFloat(counterfactualPct),
    }),
  });

  const stats = statsQuery.data;
  const graph = graphQuery.data;
  const hypotheses = hypothesesQuery.data ?? [];
  const rules = rulesQuery.data ?? [];
  const variables = variablesQuery.data ?? [];
  const ontology = ontologyQuery.data ?? [];

  const viewButtons: { key: typeof causalView; label: string; icon: any }[] = [
    { key: "overview", label: "Overview", icon: Activity },
    { key: "graph", label: "Causal Graph", icon: GitMerge },
    { key: "hypotheses", label: "Hypotheses", icon: Lightbulb },
    { key: "intervention", label: "Interventions", icon: Settings2 },
    { key: "ontology", label: "Ontology", icon: Network },
    { key: "rules", label: "Causal Rules", icon: BookOpen },
    { key: "pressure", label: "Pressure Regimes", icon: Gauge },
  ];

  return (
    <div className="space-y-4" data-testid="causal-discovery-panel">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2" data-testid="text-causal-discovery-title">
              <GitMerge className="h-5 w-5 text-purple-500" />
              Causal Physics Discovery Engine
            </CardTitle>
            <Button
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
              size="sm"
              data-testid="button-run-causal-discovery"
            >
              {runMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Play className="h-4 w-4 mr-1" />}
              Run Causal Discovery
            </Button>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Discovers causal mechanisms (not just correlations) underlying superconductivity using PC-algorithm-inspired graph inference with physics ontology constraints.
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {viewButtons.map(vb => (
              <Button
                key={vb.key}
                variant={causalView === vb.key ? "default" : "outline"}
                size="sm"
                onClick={() => setCausalView(vb.key)}
                data-testid={`button-causal-view-${vb.key}`}
              >
                <vb.icon className="h-3.5 w-3.5 mr-1" />
                {vb.label}
              </Button>
            ))}
          </div>

          {causalView === "overview" && (
            <div className="space-y-4">
              {statsQuery.isLoading ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[0,1,2,3].map(i => <Skeleton key={i} className="h-20" />)}
                </div>
              ) : stats ? (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-2xl font-bold text-purple-600" data-testid="text-causal-total-runs">{stats.totalRuns}</div>
                        <div className="text-xs text-muted-foreground">Discovery Runs</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-2xl font-bold text-blue-600" data-testid="text-causal-variables">{stats.causalVariableCount}</div>
                        <div className="text-xs text-muted-foreground">Causal Variables</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-2xl font-bold text-green-600" data-testid="text-causal-hypotheses">{stats.totalHypotheses}</div>
                        <div className="text-xs text-muted-foreground">Hypotheses</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-2xl font-bold text-amber-600" data-testid="text-causal-rules">{stats.totalCausalRules}</div>
                        <div className="text-xs text-muted-foreground">Causal Rules</div>
                      </CardContent>
                    </Card>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-lg font-bold" data-testid="text-causal-ontology-nodes">{stats.ontologyNodeCount}</div>
                        <div className="text-xs text-muted-foreground">Ontology Nodes</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-lg font-bold" data-testid="text-causal-intervention-vars">{stats.interventionVariableCount}</div>
                        <div className="text-xs text-muted-foreground">Intervention Variables</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4 pb-3 text-center">
                        <div className="text-lg font-bold" data-testid="text-causal-graph-size">
                          {stats.latestGraphSize.nodes}N / {stats.latestGraphSize.edges}E
                        </div>
                        <div className="text-xs text-muted-foreground">Latest Graph (Nodes/Edges)</div>
                      </CardContent>
                    </Card>
                  </div>

                  {stats.variableCategories && (
                    <Card>
                      <CardHeader className="pb-2"><CardTitle className="text-sm">Variable Categories</CardTitle></CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(stats.variableCategories).map(([cat, count]) => (
                            <Badge key={cat} variant="secondary" data-testid={`badge-cat-${cat}`}>
                              {cat.replace(/_/g, " ")}: {count as number}
                            </Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {stats.topEdges && stats.topEdges.length > 0 && (
                    <Card>
                      <CardHeader className="pb-2"><CardTitle className="text-sm">Top Causal Edges</CardTitle></CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {stats.topEdges.slice(0, 6).map((e: any, i: number) => (
                            <div key={i} className="flex items-center gap-2 text-sm" data-testid={`causal-edge-${i}`}>
                              <Badge variant="outline" className="font-mono text-xs">{e.source}</Badge>
                              <ArrowRight className="h-3 w-3 text-purple-500" />
                              <Badge variant="outline" className="font-mono text-xs">{e.target}</Badge>
                              <span className="text-muted-foreground ml-auto text-xs">strength: {e.strength}</span>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No causal discovery data yet. Click "Run Causal Discovery" to begin.</p>
              )}
            </div>
          )}

          {causalView === "graph" && (
            <div className="space-y-4">
              {graph && graph.edges && graph.edges.length > 0 ? (
                <>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span>Method: {graph.method}</span>
                    <span>Nodes: {graph.nodes?.length ?? 0}</span>
                    <span>Edges: {graph.edges?.length ?? 0}</span>
                    <span>Dataset: {graph.datasetSize} records</span>
                  </div>
                  <Card>
                    <CardHeader className="pb-2"><CardTitle className="text-sm">Discovered Causal Edges</CardTitle></CardHeader>
                    <CardContent>
                      <div className="max-h-96 overflow-y-auto space-y-2">
                        {graph.edges.map((e: any, i: number) => (
                          <div key={i} className="border rounded p-2 text-sm" data-testid={`graph-edge-${i}`}>
                            <div className="flex items-center gap-2 mb-1">
                              <code className="text-xs bg-muted px-1 rounded">{e.source}</code>
                              <ArrowRight className="h-3 w-3" />
                              <code className="text-xs bg-muted px-1 rounded">{e.target}</code>
                              <Badge variant={e.strength > 0.5 ? "default" : "secondary"} className="ml-auto text-xs">
                                {(e.strength).toFixed(3)}
                              </Badge>
                              {e.validated && <CheckCircle2 className="h-3 w-3 text-green-500" />}
                            </div>
                            <p className="text-xs text-muted-foreground">{e.mechanism}</p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No causal graph discovered yet. Run causal discovery first.</p>
              )}
            </div>
          )}

          {causalView === "hypotheses" && (
            <div className="space-y-4">
              {hypotheses.length > 0 ? (
                <div className="space-y-3">
                  {hypotheses.map((h: any, i: number) => (
                    <Card key={h.id || i} data-testid={`hypothesis-card-${i}`}>
                      <CardContent className="pt-4 pb-3">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1">
                            <p className="text-sm font-medium" data-testid={`text-hypothesis-${i}`}>{h.statement}</p>
                            <div className="flex items-center gap-1 mt-1">
                              {h.causalChain?.map((v: string, ci: number) => (
                                <span key={ci} className="flex items-center gap-0.5">
                                  <code className="text-xs bg-muted px-1 rounded">{v}</code>
                                  {ci < h.causalChain.length - 1 && <ArrowRight className="h-2.5 w-2.5 text-muted-foreground" />}
                                </span>
                              ))}
                            </div>
                          </div>
                          <Badge variant={h.confidence > 0.5 ? "default" : "secondary"}>
                            {(h.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">{h.mechanism?.slice(0, 200)}</p>
                        {h.testableIntervention && (
                          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                            <Microscope className="h-3 w-3 inline mr-1" />
                            {h.testableIntervention}
                          </p>
                        )}
                        {h.materialFamilies && h.materialFamilies.length > 0 && (
                          <div className="flex gap-1 mt-1">
                            {h.materialFamilies.map((f: string) => (
                              <Badge key={f} variant="outline" className="text-xs">{f}</Badge>
                            ))}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No hypotheses generated yet. Run causal discovery first.</p>
              )}
            </div>
          )}

          {causalView === "intervention" && (
            <div className="space-y-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Intervention Simulator</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                    <div>
                      <label className="text-xs text-muted-foreground block mb-1">Formula</label>
                      <input
                        className="w-full border rounded px-2 py-1.5 text-sm bg-background"
                        value={interventionFormula}
                        onChange={e => setInterventionFormula(e.target.value)}
                        data-testid="input-intervention-formula"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground block mb-1">Variable</label>
                      <select
                        className="w-full border rounded px-2 py-1.5 text-sm bg-background"
                        value={interventionVar}
                        onChange={e => setInterventionVar(e.target.value)}
                        data-testid="select-intervention-var"
                      >
                        {variables.filter((v: any) => v.isIntervention).map((v: any) => (
                          <option key={v.name} value={v.name}>{v.name} ({v.unit})</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground block mb-1">New Value</label>
                      <input
                        className="w-full border rounded px-2 py-1.5 text-sm bg-background"
                        value={interventionValue}
                        onChange={e => setInterventionValue(e.target.value)}
                        type="number"
                        data-testid="input-intervention-value"
                      />
                    </div>
                  </div>
                  <Button
                    onClick={() => interventionMutation.mutate()}
                    disabled={interventionMutation.isPending}
                    size="sm"
                    data-testid="button-run-intervention"
                  >
                    {interventionMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Zap className="h-4 w-4 mr-1" />}
                    Simulate Intervention
                  </Button>

                  {interventionMutation.data && (
                    <div className="mt-4 border rounded p-3 space-y-2" data-testid="intervention-result">
                      {(() => {
                        const r: any = interventionMutation.data;
                        return (
                          <>
                            <div className="flex items-center gap-4 text-sm">
                              <span>Tc: {r.tcOriginal}K</span>
                              <ArrowRight className="h-3 w-3" />
                              <span className={r.tcChange > 0 ? "text-green-600 font-medium" : r.tcChange < 0 ? "text-red-600 font-medium" : ""}>
                                {r.tcNew}K ({r.tcChange > 0 ? "+" : ""}{r.tcChange}K)
                              </span>
                            </div>
                            {r.causalPathway && r.causalPathway.length > 1 && (
                              <div className="text-xs text-muted-foreground">
                                Causal pathway: {r.causalPathway.join(" -> ")}
                              </div>
                            )}
                            {r.effects && r.effects.length > 0 && (
                              <div className="space-y-1 mt-2">
                                <p className="text-xs font-medium">Propagated Effects:</p>
                                {r.effects.slice(0, 8).map((eff: any, i: number) => (
                                  <div key={i} className="text-xs flex items-center gap-2" data-testid={`effect-${i}`}>
                                    <code className="bg-muted px-1 rounded">{eff.variable}</code>
                                    <span>{eff.originalValue?.toFixed(2)}</span>
                                    <ArrowRight className="h-2.5 w-2.5" />
                                    <span>{eff.newValue?.toFixed(2)}</span>
                                    <span className={eff.changePercent > 0 ? "text-green-600" : "text-red-600"}>
                                      ({eff.changePercent > 0 ? "+" : ""}{eff.changePercent}%)
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </>
                        );
                      })()}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Counterfactual Simulator</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                    <div>
                      <label className="text-xs text-muted-foreground block mb-1">Variable</label>
                      <select
                        className="w-full border rounded px-2 py-1.5 text-sm bg-background"
                        value={counterfactualVar}
                        onChange={e => setCounterfactualVar(e.target.value)}
                        data-testid="select-counterfactual-var"
                      >
                        {variables.map((v: any) => (
                          <option key={v.name} value={v.name}>{v.name}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground block mb-1">Modification %</label>
                      <input
                        className="w-full border rounded px-2 py-1.5 text-sm bg-background"
                        value={counterfactualPct}
                        onChange={e => setCounterfactualPct(e.target.value)}
                        type="number"
                        data-testid="input-counterfactual-pct"
                      />
                    </div>
                    <div className="flex items-end">
                      <Button
                        onClick={() => counterfactualMutation.mutate()}
                        disabled={counterfactualMutation.isPending}
                        size="sm"
                        data-testid="button-run-counterfactual"
                      >
                        {counterfactualMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Microscope className="h-4 w-4 mr-1" />}
                        Ask "What If?"
                      </Button>
                    </div>
                  </div>

                  {counterfactualMutation.data && (
                    <div className="mt-3 border rounded p-3 space-y-2" data-testid="counterfactual-result">
                      {(() => {
                        const r: any = counterfactualMutation.data;
                        return (
                          <>
                            <p className="text-sm font-medium">{r.question}</p>
                            <div className="flex items-center gap-4 text-sm">
                              <span>Tc: {r.originalTc?.toFixed(1)}K</span>
                              <ArrowRight className="h-3 w-3" />
                              <span className={r.tcDelta > 0 ? "text-green-600 font-medium" : r.tcDelta < 0 ? "text-red-600 font-medium" : ""}>
                                {r.counterfactualTc?.toFixed(1)}K ({r.tcDelta > 0 ? "+" : ""}{r.tcDelta?.toFixed(1)}K, {r.tcDeltaPercent > 0 ? "+" : ""}{r.tcDeltaPercent}%)
                              </span>
                            </div>
                            <p className="text-xs text-blue-600 dark:text-blue-400">{r.designImplication}</p>
                            {r.propagatedEffects && r.propagatedEffects.length > 0 && (
                              <div className="mt-2">
                                <p className="text-xs font-medium">Downstream effects:</p>
                                <div className="flex flex-wrap gap-2 mt-1">
                                  {r.propagatedEffects.slice(0, 6).map((pe: any, i: number) => (
                                    <Badge key={i} variant="outline" className="text-xs">
                                      {pe.variable}: {pe.change > 0 ? "+" : ""}{pe.change?.toFixed(1)}%
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </>
                        );
                      })()}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {causalView === "ontology" && (
            <div className="space-y-4">
              {ontology.length > 0 ? (
                <>
                  {["atomic_structure", "electronic_structure", "phonon_properties", "pairing_interactions", "thermodynamic_conditions", "superconducting_properties", "topological"].map(cat => {
                    const nodes = ontology.filter((n: any) => n.category === cat);
                    if (nodes.length === 0) return null;
                    return (
                      <Card key={cat}>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm capitalize">{cat.replace(/_/g, " ")}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            {nodes.map((n: any) => (
                              <div key={n.variable} className="border rounded p-2 text-xs" data-testid={`ontology-node-${n.variable}`}>
                                <div className="flex items-center justify-between mb-1">
                                  <code className="font-medium">{n.variable}</code>
                                  <Badge variant="secondary">Level {n.level}</Badge>
                                </div>
                                {n.parents.length > 0 && (
                                  <div className="text-muted-foreground">
                                    Parents: {n.parents.map((p: string) => <code key={p} className="bg-muted px-0.5 rounded mr-1">{p}</code>)}
                                  </div>
                                )}
                                {n.children.length > 0 && (
                                  <div className="text-muted-foreground mt-0.5">
                                    Children: {n.children.map((c: string) => <code key={c} className="bg-muted px-0.5 rounded mr-1">{c}</code>)}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">Loading ontology...</p>
              )}
            </div>
          )}

          {causalView === "rules" && (
            <div className="space-y-3">
              {rules.length > 0 ? (
                rules.map((r: any, i: number) => (
                  <Card key={i} data-testid={`causal-rule-${i}`}>
                    <CardContent className="pt-4 pb-3">
                      <div className="flex items-center gap-2 text-sm mb-1">
                        <Badge variant="outline">{r.antecedent}</Badge>
                        <ArrowRight className="h-3 w-3 text-purple-500" />
                        <Badge variant="outline">{r.consequent}</Badge>
                        <span className="ml-auto text-xs text-muted-foreground">strength: {r.strength?.toFixed(3)}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">{r.mechanism}</p>
                      {r.validatedAcross && r.validatedAcross.length > 0 && (
                        <div className="flex gap-1 mt-1">
                          <span className="text-xs text-muted-foreground">Validated:</span>
                          {r.validatedAcross.map((f: string) => (
                            <Badge key={f} variant="secondary" className="text-xs">{f}</Badge>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No causal rules extracted yet. Run causal discovery first.</p>
              )}
            </div>
          )}

          {causalView === "pressure" && (
            <div className="space-y-4">
              {runMutation.data ? (
                (() => {
                  const pc: any = (runMutation.data as any).pressureComparison;
                  if (!pc) return <p className="text-sm text-muted-foreground text-center py-8">No pressure comparison data. Run discovery first.</p>;
                  return (
                    <>
                      <Card>
                        <CardHeader className="pb-2"><CardTitle className="text-sm">Decompression Insight</CardTitle></CardHeader>
                        <CardContent>
                          <p className="text-sm" data-testid="text-decompression-insight">{pc.decompressionInsight}</p>
                        </CardContent>
                      </Card>

                      {pc.survivingMechanisms && pc.survivingMechanisms.length > 0 && (
                        <Card>
                          <CardHeader className="pb-2"><CardTitle className="text-sm">Mechanisms Surviving Across Pressure Regimes</CardTitle></CardHeader>
                          <CardContent>
                            <div className="space-y-2">
                              {pc.survivingMechanisms.map((m: any, i: number) => (
                                <div key={i} className="flex items-center gap-2 text-sm border rounded p-2" data-testid={`surviving-mechanism-${i}`}>
                                  <code className="text-xs">{m.source}</code>
                                  <ArrowRight className="h-3 w-3" />
                                  <code className="text-xs">{m.target}</code>
                                  <span className="ml-auto text-xs text-muted-foreground">
                                    ambient: {m.ambientStrength?.toFixed(3)} | high-P: {m.hpStrength?.toFixed(3)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {pc.newMechanisms && pc.newMechanisms.length > 0 && (
                        <Card>
                          <CardHeader className="pb-2"><CardTitle className="text-sm">Regime-Specific Mechanisms</CardTitle></CardHeader>
                          <CardContent>
                            <div className="space-y-2">
                              {pc.newMechanisms.map((m: any, i: number) => (
                                <div key={i} className="flex items-center gap-2 text-sm" data-testid={`new-mechanism-${i}`}>
                                  <Badge variant={m.regime === "ambient" ? "default" : "secondary"}>{m.regime}</Badge>
                                  <code className="text-xs">{m.source}</code>
                                  <ArrowRight className="h-3 w-3" />
                                  <code className="text-xs">{m.target}</code>
                                  <span className="ml-auto text-xs">{m.strength?.toFixed(3)}</span>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </>
                  );
                })()
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">Run causal discovery to see pressure regime comparison data.</p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
