import { useQuery } from "@tanstack/react-query";
import type { SuperconductorCandidate, ComputationalResult, CrystalStructure } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Atom, Zap, Filter, XCircle, CheckCircle2,
  Activity, Layers, Magnet, Gauge, Target,
  Beaker, ArrowDown,
} from "lucide-react";

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

function PhysicsPropertyCard({ candidate }: { candidate: SuperconductorCandidate }) {
  const competingPhases = (candidate.competingPhases as any[]) ?? [];

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
  if (value == null) return null;
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
              <span className="font-mono font-bold">{lattice.a?.toFixed(2)}</span>
            </div>
            <div className="p-1 bg-muted/50 rounded text-center">
              <span className="text-muted-foreground">b=</span>
              <span className="font-mono font-bold">{lattice.b?.toFixed(2)}</span>
            </div>
            <div className="p-1 bg-muted/50 rounded text-center">
              <span className="text-muted-foreground">c=</span>
              <span className="font-mono font-bold">{lattice.c?.toFixed(2)}</span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 text-xs">
          {structure.decompositionEnergy != null && (
            <div>
              <span className="text-muted-foreground">Hull dist: </span>
              <span className="font-mono">{structure.convexHullDistance?.toFixed(3)} eV/atom</span>
            </div>
          )}
          {structure.synthesizability != null && (
            <div>
              <span className="text-muted-foreground">Synth: </span>
              <span className="font-mono">{(structure.synthesizability * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
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

  const { data: failedData } = useQuery<{ results: ComputationalResult[] }>({
    queryKey: ["/api/computational-results/failed"],
  });

  const { data: structureData } = useQuery<{ structures: CrystalStructure[]; total: number }>({
    queryKey: ["/api/crystal-structures"],
  });

  const candidates = scData?.candidates ?? [];
  const physicsAnalyzed = candidates.filter(c => c.electronPhononCoupling != null || c.verificationStage != null && c.verificationStage > 0);
  const failedResults = failedData?.results ?? [];
  const structures = structureData?.structures ?? [];

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
          <TabsTrigger value="structures" data-testid="tab-structures">Crystal Structures</TabsTrigger>
          <TabsTrigger value="failures" data-testid="tab-failures">Negative Results</TabsTrigger>
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
              {physicsAnalyzed.map(c => <PhysicsPropertyCard key={c.id} candidate={c} />)}
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
      </Tabs>
    </div>
  );
}
