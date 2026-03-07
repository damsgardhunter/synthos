import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { queryClient } from "@/lib/queryClient";
import { useWebSocket } from "@/hooks/use-websocket";
import type { SuperconductorCandidate, ComputationalResult, CrystalStructure } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Link } from "wouter";
import {
  Atom, Zap, Filter, XCircle, CheckCircle2,
  Activity, Layers, Magnet, Gauge, Target,
  Beaker, ArrowDown, ExternalLink, Thermometer,
  FlaskConical, Star, Bug, Brain, Diamond, ClipboardList,
} from "lucide-react";

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

      <Tabs defaultValue="pipeline" className="space-y-4">
        <TabsList data-testid="physics-tabs">
          <TabsTrigger value="pipeline" data-testid="tab-pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="physics" data-testid="tab-physics">Physics Properties</TabsTrigger>
          <TabsTrigger value="dft-selections" data-testid="tab-dft-selections">DFT Selections</TabsTrigger>
          <TabsTrigger value="structures" data-testid="tab-structures">Crystal Structures</TabsTrigger>
          <TabsTrigger value="failures" data-testid="tab-failures">Negative Results</TabsTrigger>
          <TabsTrigger value="synthesis" data-testid="tab-synthesis">Synthesis Variables</TabsTrigger>
          <TabsTrigger value="advanced-physics" data-testid="tab-advanced-physics">Advanced Physics</TabsTrigger>
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
      </Tabs>
    </div>
  );
}
