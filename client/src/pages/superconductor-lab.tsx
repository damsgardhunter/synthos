import { useQuery } from "@tanstack/react-query";
import type { SuperconductorCandidate, SynthesisProcess, ChemicalReaction } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Zap, Thermometer, Shield, Atom, FlaskConical, Beaker, Target,
  CheckCircle2, XCircle, ArrowRight, Gauge, Magnet,
} from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  "theoretical": "bg-gray-100 text-gray-700 dark:bg-gray-900 dark:text-gray-300",
  "promising": "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
  "novel-design": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  "high-tc-candidate": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  "under-review": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  "requires-verification": "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  "validated": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
};

const DIFFICULTY_COLORS: Record<string, string> = {
  "easy": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
  "moderate": "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
  "hard": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  "extreme": "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
};

function BoolIndicator({ value, label }: { value: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      {value ? (
        <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
      ) : (
        <XCircle className="h-3.5 w-3.5 text-muted-foreground/40" />
      )}
      <span className={`text-xs ${value ? "text-foreground" : "text-muted-foreground/60"}`}>{label}</span>
    </div>
  );
}

function ScoreBar({ label, score, color }: { label: string; score: number | null; color: string }) {
  const pct = (score ?? 0) * 100;
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-xs font-mono font-bold">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function CandidateCard({ candidate }: { candidate: SuperconductorCandidate }) {
  const statusColor = STATUS_COLORS[candidate.status] ?? STATUS_COLORS["theoretical"];
  const tcColor = (candidate.predictedTc ?? 0) >= 293 ? "text-green-600 dark:text-green-400" : "text-foreground";

  return (
    <Card data-testid={`sc-candidate-${candidate.id}`} className="flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <div>
            <CardTitle className="text-base leading-snug">{candidate.name}</CardTitle>
            <p className="text-sm font-mono text-primary mt-1">{candidate.formula}</p>
          </div>
          <Badge className={`${statusColor} border-0`}>{candidate.status}</Badge>
        </div>
      </CardHeader>
      <CardContent className="flex-1 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="p-2.5 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1.5 mb-0.5">
              <Thermometer className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Predicted Tc</span>
            </div>
            <p className={`text-lg font-mono font-bold ${tcColor}`}>
              {candidate.predictedTc ? `${candidate.predictedTc}K` : "N/A"}
            </p>
          </div>
          <div className="p-2.5 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1.5 mb-0.5">
              <Gauge className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Pressure</span>
            </div>
            <p className="text-lg font-mono font-bold">
              {candidate.pressureGpa != null ? `${candidate.pressureGpa} GPa` : "Ambient"}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 p-2.5 bg-muted/30 rounded-md">
          <BoolIndicator value={candidate.meissnerEffect ?? false} label="Meissner Effect" />
          <BoolIndicator value={candidate.zeroResistance ?? false} label="Zero Resistance" />
          <BoolIndicator value={candidate.roomTempViable ?? false} label="Room Temp Viable" />
          <BoolIndicator value={(candidate.quantumCoherence ?? 0) > 0.5} label="Quantum Coherent" />
        </div>

        {(candidate.electronPhononCoupling || candidate.correlationStrength || candidate.upperCriticalField) && (
          <div className="grid grid-cols-3 gap-1.5 text-xs">
            {candidate.electronPhononCoupling != null && (
              <div className="p-1.5 bg-blue-50 dark:bg-blue-950/30 rounded">
                <span className="text-muted-foreground block text-[10px]">e-ph lambda</span>
                <span className="font-mono font-bold">{candidate.electronPhononCoupling.toFixed(2)}</span>
              </div>
            )}
            {candidate.correlationStrength != null && (
              <div className="p-1.5 bg-purple-50 dark:bg-purple-950/30 rounded">
                <span className="text-muted-foreground block text-[10px]">U/W ratio</span>
                <span className="font-mono font-bold">{candidate.correlationStrength.toFixed(2)}</span>
              </div>
            )}
            {candidate.upperCriticalField != null && (
              <div className="p-1.5 bg-green-50 dark:bg-green-950/30 rounded">
                <span className="text-muted-foreground block text-[10px]">Hc2</span>
                <span className="font-mono font-bold">{candidate.upperCriticalField.toFixed(1)}T</span>
              </div>
            )}
          </div>
        )}

        {(candidate.pairingMechanism || candidate.dimensionality || candidate.pairingSymmetry) && (
          <div className="flex flex-wrap gap-1">
            {candidate.pairingMechanism && <Badge variant="outline" className="text-[10px]">{candidate.pairingMechanism}</Badge>}
            {candidate.pairingSymmetry && <Badge variant="outline" className="text-[10px]">{candidate.pairingSymmetry}</Badge>}
            {candidate.dimensionality && <Badge variant="outline" className="text-[10px]">{candidate.dimensionality}</Badge>}
          </div>
        )}

        {candidate.cooperPairMechanism && (
          <div className="text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">Cooper Pairs: </span>
            {candidate.cooperPairMechanism}
          </div>
        )}

        <div className="space-y-1.5">
          <ScoreBar label="XGBoost" score={candidate.xgboostScore} color="bg-blue-500" />
          <ScoreBar label="Neural Net" score={candidate.neuralNetScore} color="bg-purple-500" />
          <ScoreBar label="Ensemble" score={candidate.ensembleScore} color="bg-primary" />
        </div>

        {candidate.uncertaintyEstimate != null && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Uncertainty:</span>
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${candidate.uncertaintyEstimate > 0.6 ? "bg-red-500" : candidate.uncertaintyEstimate > 0.3 ? "bg-yellow-500" : "bg-green-500"}`}
                style={{ width: `${(candidate.uncertaintyEstimate ?? 0) * 100}%` }}
              />
            </div>
            <span className="font-mono text-[10px]">{((candidate.uncertaintyEstimate ?? 0) * 100).toFixed(0)}%</span>
          </div>
        )}

        {candidate.verificationStage != null && candidate.verificationStage > 0 && (
          <div className="flex items-center gap-1 text-xs">
            <span className="text-muted-foreground">Pipeline Stage:</span>
            <div className="flex gap-0.5">
              {[0,1,2,3,4].map(s => (
                <div key={s} className={`h-2 w-4 rounded-sm ${s <= (candidate.verificationStage ?? 0) ? "bg-primary" : "bg-muted"}`} />
              ))}
            </div>
            <span className="font-mono text-[10px]">{candidate.verificationStage}/4</span>
          </div>
        )}

        {candidate.notes && (
          <p className="text-xs text-muted-foreground leading-relaxed border-t border-border pt-2">{candidate.notes}</p>
        )}
      </CardContent>
    </Card>
  );
}

function SynthesisCard({ process }: { process: SynthesisProcess }) {
  const diffColor = DIFFICULTY_COLORS[process.difficulty] ?? DIFFICULTY_COLORS["moderate"];
  const conditions = process.conditions as Record<string, any> ?? {};
  const steps = process.steps ?? [];
  const precursors = process.precursors ?? [];
  const equipment = process.equipment ?? [];

  return (
    <Card data-testid={`synthesis-${process.id}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <div>
            <CardTitle className="text-base leading-snug">{process.materialName}</CardTitle>
            <p className="text-sm font-mono text-primary mt-1">{process.formula}</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={`${diffColor} border-0`}>{process.difficulty}</Badge>
            <Badge variant="outline">{process.method}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {(conditions.temperature || conditions.pressure || conditions.atmosphere || conditions.holdTime) && (
          <div className="space-y-2">
            <div className="grid grid-cols-3 gap-2 text-xs">
              {conditions.temperature != null && (
                <div className="p-2 bg-muted/50 rounded-md">
                  <span className="text-muted-foreground block">Peak Temp</span>
                  <span className="font-mono font-bold">{conditions.temperature}C</span>
                </div>
              )}
              {conditions.pressure != null && (
                <div className="p-2 bg-muted/50 rounded-md">
                  <span className="text-muted-foreground block">Pressure</span>
                  <span className="font-mono font-bold">{conditions.pressure} atm</span>
                </div>
              )}
              {conditions.atmosphere && (
                <div className="p-2 bg-muted/50 rounded-md">
                  <span className="text-muted-foreground block">Atmosphere</span>
                  <span className="font-mono font-bold">{conditions.atmosphere}</span>
                </div>
              )}
            </div>
            {(conditions.heatingRate || conditions.holdTime || conditions.coolingMethod) && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                {conditions.heatingRate && (
                  <div className="p-2 bg-muted/30 rounded-md">
                    <span className="text-muted-foreground block">Heating Rate</span>
                    <span className="font-mono text-xs">{conditions.heatingRate}</span>
                  </div>
                )}
                {conditions.holdTime && (
                  <div className="p-2 bg-muted/30 rounded-md">
                    <span className="text-muted-foreground block">Hold Time</span>
                    <span className="font-mono text-xs">{conditions.holdTime}</span>
                  </div>
                )}
                {conditions.coolingMethod && (
                  <div className="p-2 bg-muted/30 rounded-md">
                    <span className="text-muted-foreground block">Cooling</span>
                    <span className="font-mono text-xs">{conditions.coolingMethod}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {precursors.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Precursors</p>
            <div className="flex flex-wrap gap-1">
              {precursors.map((p, i) => (
                <Badge key={i} variant="secondary" className="text-xs font-mono">{p}</Badge>
              ))}
            </div>
          </div>
        )}

        {steps.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Procedure</p>
            <ol className="space-y-1">
              {steps.map((step, i) => (
                <li key={i} className="flex gap-2 text-xs text-muted-foreground">
                  <span className="text-primary font-bold min-w-[1rem]">{i + 1}.</span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        <div className="flex items-center gap-4 text-xs text-muted-foreground border-t border-border pt-2">
          {process.timeEstimate && <span>Duration: {process.timeEstimate}</span>}
          {process.yieldPercent != null && <span>Yield: {process.yieldPercent}%</span>}
          {equipment.length > 0 && <span>Equipment: {equipment.length} items</span>}
        </div>

        {process.safetyNotes && (
          <div className="flex items-start gap-1.5 p-2 bg-red-50 dark:bg-red-950/30 rounded-md text-xs text-red-700 dark:text-red-400">
            <Shield className="h-3 w-3 mt-0.5 flex-shrink-0" />
            <span>{process.safetyNotes}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ReactionCard({ reaction }: { reaction: ChemicalReaction }) {
  const reactants = (reaction.reactants ?? []) as { formula?: string; role?: string }[];
  const products = (reaction.products ?? []) as { formula?: string; role?: string }[];
  const conditions = reaction.conditions as Record<string, any> ?? {};
  const energetics = (reaction.energetics ?? {}) as Record<string, any>;
  const relevance = reaction.relevanceToSuperconductor ?? 0;

  return (
    <Card data-testid={`reaction-${reaction.id}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <CardTitle className="text-base leading-snug">{reaction.name}</CardTitle>
          <Badge variant="outline">{reaction.reactionType}</Badge>
        </div>
        <p className="text-sm font-mono text-primary bg-muted/50 p-2 rounded-md mt-2">{reaction.equation}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          {reactants.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-muted-foreground mb-1">Reactants</p>
              <div className="space-y-0.5">
                {reactants.map((r, i) => (
                  <div key={i} className="text-xs">
                    <span className="font-mono font-medium">{r.formula ?? String(r)}</span>
                    {r.role && <span className="text-muted-foreground ml-1">({r.role})</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
          {products.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-muted-foreground mb-1">Products</p>
              <div className="space-y-0.5">
                {products.map((p, i) => (
                  <div key={i} className="text-xs">
                    <span className="font-mono font-medium">{p.formula ?? String(p)}</span>
                    {p.role && <span className="text-muted-foreground ml-1">({p.role})</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {(conditions.temperature || conditions.pressure || conditions.catalyst) && (
          <div className="flex flex-wrap gap-2 text-xs">
            {conditions.temperature && <Badge variant="secondary">T: {conditions.temperature}</Badge>}
            {conditions.pressure && <Badge variant="secondary">P: {conditions.pressure}</Badge>}
            {conditions.catalyst && <Badge variant="secondary">Cat: {conditions.catalyst}</Badge>}
            {conditions.atmosphere && <Badge variant="secondary">{conditions.atmosphere}</Badge>}
          </div>
        )}

        {(energetics.deltaH != null || energetics.deltaG != null) && (
          <div className="grid grid-cols-3 gap-2 text-xs">
            {energetics.deltaH != null && (
              <div className="p-2 bg-muted/50 rounded-md">
                <span className="text-muted-foreground block">delta-H</span>
                <span className="font-mono font-bold">{energetics.deltaH} kJ/mol</span>
              </div>
            )}
            {energetics.deltaG != null && (
              <div className="p-2 bg-muted/50 rounded-md">
                <span className="text-muted-foreground block">delta-G</span>
                <span className="font-mono font-bold">{energetics.deltaG} kJ/mol</span>
              </div>
            )}
            {energetics.activationEnergy != null && (
              <div className="p-2 bg-muted/50 rounded-md">
                <span className="text-muted-foreground block">Ea</span>
                <span className="font-mono font-bold">{energetics.activationEnergy} kJ/mol</span>
              </div>
            )}
          </div>
        )}

        {reaction.mechanism && (
          <p className="text-xs text-muted-foreground leading-relaxed">{reaction.mechanism}</p>
        )}

        {relevance > 0 && (
          <div className="space-y-0.5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">SC Relevance</span>
              <span className="text-xs font-mono font-bold">{(relevance * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${relevance > 0.7 ? "bg-green-500" : relevance > 0.4 ? "bg-yellow-500" : "bg-muted-foreground/30"}`}
                style={{ width: `${relevance * 100}%` }}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function SuperconductorLab() {
  const { data: scData, isLoading: scLoading } = useQuery<{ candidates: SuperconductorCandidate[]; total: number }>({
    queryKey: ["/api/superconductor-candidates"],
  });

  const { data: synthData, isLoading: synthLoading } = useQuery<{ processes: SynthesisProcess[]; total: number }>({
    queryKey: ["/api/synthesis-processes"],
  });

  const { data: rxnData, isLoading: rxnLoading } = useQuery<{ reactions: ChemicalReaction[]; total: number }>({
    queryKey: ["/api/chemical-reactions"],
  });

  const candidates = scData?.candidates ?? [];
  const processes = synthData?.processes ?? [];
  const reactions = rxnData?.reactions ?? [];

  const roomTempCandidates = candidates.filter(c => c.roomTempViable);
  const meissnerCandidates = candidates.filter(c => c.meissnerEffect);
  const highScoreCandidates = candidates.filter(c => (c.ensembleScore ?? 0) > 0.6);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2" data-testid="text-page-title">
          <Magnet className="h-6 w-6 text-primary" />
          Superconductor Lab
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          XGBoost + Neural Network hybrid ML engine targeting room-temperature superconductivity.
          Meissner effect, Cooper pairs, zero resistance, quantum coherence analysis.
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Card data-testid="stat-total-candidates">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <Target className="h-4 w-4 text-primary" />
              <span className="text-xs text-muted-foreground">Candidates</span>
            </div>
            <div className="text-2xl font-bold font-mono">{scData?.total ?? 0}</div>
          </CardContent>
        </Card>
        <Card data-testid="stat-room-temp">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <Thermometer className="h-4 w-4 text-green-500" />
              <span className="text-xs text-muted-foreground">Room Temp</span>
            </div>
            <div className="text-2xl font-bold font-mono">{roomTempCandidates.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="stat-synthesis">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <FlaskConical className="h-4 w-4 text-purple-500" />
              <span className="text-xs text-muted-foreground">Synthesis Paths</span>
            </div>
            <div className="text-2xl font-bold font-mono">{synthData?.total ?? 0}</div>
          </CardContent>
        </Card>
        <Card data-testid="stat-reactions">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <Beaker className="h-4 w-4 text-orange-500" />
              <span className="text-xs text-muted-foreground">Reactions</span>
            </div>
            <div className="text-2xl font-bold font-mono">{rxnData?.total ?? 0}</div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="candidates" className="space-y-4">
        <TabsList data-testid="tabs-sc-lab">
          <TabsTrigger value="candidates" data-testid="tab-candidates">
            <Target className="h-3.5 w-3.5 mr-1.5" />
            SC Candidates ({candidates.length})
          </TabsTrigger>
          <TabsTrigger value="synthesis" data-testid="tab-synthesis">
            <FlaskConical className="h-3.5 w-3.5 mr-1.5" />
            Synthesis ({processes.length})
          </TabsTrigger>
          <TabsTrigger value="reactions" data-testid="tab-reactions">
            <Beaker className="h-3.5 w-3.5 mr-1.5" />
            Reactions ({reactions.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="candidates" className="space-y-4">
          {roomTempCandidates.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Thermometer className="h-4 w-4 text-green-500" />
                <h2 className="text-base font-semibold">Room-Temperature Viable</h2>
                <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0">
                  {roomTempCandidates.length}
                </Badge>
              </div>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {roomTempCandidates.map(c => <CandidateCard key={c.id} candidate={c} />)}
              </div>
            </div>
          )}

          {scLoading ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {[...Array(6)].map((_, i) => <Skeleton key={i} className="h-72" />)}
            </div>
          ) : candidates.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Atom className="h-4 w-4 text-primary" />
                <h2 className="text-base font-semibold">All Candidates</h2>
                <Badge variant="secondary">{candidates.length}</Badge>
              </div>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {candidates.map(c => <CandidateCard key={c.id} candidate={c} />)}
              </div>
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center space-y-2">
                <Target className="h-8 w-8 mx-auto text-muted-foreground/40" />
                <p className="text-muted-foreground text-sm">No superconductor candidates yet</p>
                <p className="text-muted-foreground text-xs">Start the learning engine to begin ML-powered SC research</p>
              </CardContent>
            </Card>
          )}

          <Card className="border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                ML Ensemble Architecture
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="h-6 w-6 rounded-full bg-blue-500/10 text-blue-500 text-xs font-bold flex items-center justify-center">1</div>
                    <span className="text-sm font-medium">XGBoost Feature Layer</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed pl-8">
                    Gradient boosted decision trees extract 18 physics-informed features: electronegativity spread, Cooper pair strength, phonon coupling, Meissner potential, d-wave symmetry
                  </p>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="h-6 w-6 rounded-full bg-purple-500/10 text-purple-500 text-xs font-bold flex items-center justify-center">2</div>
                    <span className="text-sm font-medium">Neural Network Refinement</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed pl-8">
                    Deep reasoning about BCS theory, electron-phonon coupling, magnetic flux expulsion, and quantum decoherence mechanisms to refine Tc predictions
                  </p>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="h-6 w-6 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center">3</div>
                    <span className="text-sm font-medium">Ensemble Scoring</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed pl-8">
                    XGBoost (40%) + Neural Network (60%) weighted ensemble with room-temperature viability assessment and synthesis pathway generation
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="synthesis" className="space-y-4">
          {synthLoading ? (
            <div className="grid gap-4 md:grid-cols-2">
              {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-64" />)}
            </div>
          ) : processes.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2">
              {processes.map(p => <SynthesisCard key={p.id} process={p} />)}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center space-y-2">
                <FlaskConical className="h-8 w-8 mx-auto text-muted-foreground/40" />
                <p className="text-muted-foreground text-sm">No synthesis processes discovered yet</p>
                <p className="text-muted-foreground text-xs">The engine maps how every material is created in a lab</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="reactions" className="space-y-4">
          {rxnLoading ? (
            <div className="grid gap-4 md:grid-cols-2">
              {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-48" />)}
            </div>
          ) : reactions.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2">
              {reactions.map(r => <ReactionCard key={r.id} reaction={r} />)}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center space-y-2">
                <Beaker className="h-8 w-8 mx-auto text-muted-foreground/40" />
                <p className="text-muted-foreground text-sm">No chemical reactions catalogued yet</p>
                <p className="text-muted-foreground text-xs">The engine learns reactions critical to superconductor creation</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
