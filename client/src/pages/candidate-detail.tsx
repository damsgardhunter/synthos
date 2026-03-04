import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRoute, Link } from "wouter";
import type { SuperconductorCandidate, CrystalStructure, ComputationalResult, SynthesisProcess, ChemicalReaction } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { safeNum, safeDisplay } from "@/lib/utils";
import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  ArrowLeft, Thermometer, Gauge, CheckCircle2, XCircle,
  Atom, Layers, Beaker, FlaskConical, Activity, Magnet,
  Shield, Target, Zap, Database, AlertTriangle, ExternalLink,
  ClipboardCheck, Plus,
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

function TcConfidenceRange({ predictedTc, p90 }: { predictedTc: number; p90: number }) {
  const band = computeConfidenceBand(predictedTc, p90);
  return (
    <div className="flex items-center gap-1.5 mt-1" data-testid="tc-confidence-range">
      <span className="text-[10px] text-muted-foreground">90% CI:</span>
      <span className="text-xs font-mono text-muted-foreground">{band.lower}K</span>
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden relative min-w-[50px]">
        <div
          className="absolute inset-y-0 bg-primary/25 rounded-full"
          style={{
            left: `${Math.min(95, (band.lower / Math.max(band.upper, 1)) * 100)}%`,
            right: "0%",
          }}
        />
        <div
          className="absolute inset-y-0 w-0.5 bg-primary"
          style={{
            left: `${Math.min(95, (predictedTc / Math.max(band.upper, 1)) * 100)}%`,
          }}
        />
      </div>
      <span className="text-xs font-mono text-muted-foreground">{band.upper}K</span>
    </div>
  );
}
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

function ConfidenceBadge({ level }: { level?: string | null }) {
  if (!level) return null;
  const styles: Record<string, string> = {
    high: "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-400",
    medium: "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-400",
    low: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400",
  };
  const labels: Record<string, string> = { high: "DFT", medium: "Model", low: "Est." };
  return (
    <span data-testid={`confidence-badge-${level}`} className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${styles[level] ?? styles.low}`}>
      {labels[level] ?? level}
    </span>
  );
}

interface CandidateProfile {
  formula: string;
  candidates: SuperconductorCandidate[];
  crystalStructures: CrystalStructure[];
  computationalResults: ComputationalResult[];
  synthesisProcesses: SynthesisProcess[];
  chemicalReactions: ChemicalReaction[];
}

const STATUS_COLORS: Record<string, string> = {
  "theoretical": "bg-gray-100 text-gray-700 dark:bg-gray-900 dark:text-gray-300",
  "promising": "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
  "novel-design": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  "high-tc-candidate": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  "under-review": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  "requires-verification": "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  "validated": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
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
  const pct = safeNum(score) * 100;
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-xs font-mono font-bold">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function CandidateHeader({ candidate, p90 }: { candidate: SuperconductorCandidate; p90?: number }) {
  const statusColor = STATUS_COLORS[candidate.status] ?? STATUS_COLORS["theoretical"];
  const tcColor = (candidate.predictedTc ?? 0) >= 293 ? "text-green-600 dark:text-green-400" : "text-foreground";

  return (
    <Card data-testid="candidate-header">
      <CardContent className="pt-6 space-y-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h2 className="text-xl font-bold">{candidate.name}</h2>
            <p className="text-lg font-mono text-primary mt-1">{candidate.formula}</p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Badge className={`${statusColor} border-0`}>{candidate.status}</Badge>
            {candidate.verificationStage != null && candidate.verificationStage > 0 && (
              <Badge variant="outline">Stage {candidate.verificationStage}/5</Badge>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1.5 mb-1">
              <Thermometer className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Predicted Tc</span>
            </div>
            <div className="flex items-center gap-1.5">
              <p className={`text-xl font-mono font-bold ${tcColor}`}>
                {candidate.predictedTc ? `${candidate.predictedTc}K` : "N/A"}
              </p>
              <ConfidenceBadge level={candidate.dataConfidence} />
            </div>
            {candidate.predictedTc != null && p90 != null && (
              <TcConfidenceRange predictedTc={candidate.predictedTc} p90={p90} />
            )}
          </div>
          <div className="p-3 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1.5 mb-1">
              <Gauge className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Pressure</span>
            </div>
            <p className="text-xl font-mono font-bold">
              {candidate.pressureGpa != null
                ? (candidate.pressureGpa <= 0 ? "Ambient (0 GPa)" : `${candidate.pressureGpa} GPa`)
                : "Not specified"}
            </p>
          </div>
          <div className="p-3 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1.5 mb-1">
              <Shield className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Stability</span>
            </div>
            <p className="text-xl font-mono font-bold">
              {candidate.stabilityScore != null ? safeDisplay(candidate.stabilityScore, 2) : "N/A"}
            </p>
          </div>
          <div className="p-3 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Uncertainty</span>
            </div>
            <p className="text-xl font-mono font-bold">
              {candidate.uncertaintyEstimate != null ? `${(safeNum(candidate.uncertaintyEstimate) * 100).toFixed(0)}%` : "N/A"}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-1.5 p-3 bg-muted/30 rounded-md">
          <BoolIndicator value={candidate.meissnerEffect ?? false} label="Meissner Effect" />
          <BoolIndicator value={candidate.zeroResistance ?? false} label="Zero Resistance" />
          <BoolIndicator value={candidate.roomTempViable ?? false} label="Room Temp Viable" />
          <BoolIndicator value={candidate.ambientPressureStable ?? false} label="Ambient Pressure" />
        </div>
      </CardContent>
    </Card>
  );
}

function MLScoresSection({ candidate }: { candidate: SuperconductorCandidate }) {
  return (
    <Card data-testid="ml-scores">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Activity className="h-5 w-5" />
          ML Prediction Scores
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <ScoreBar label="XGBoost" score={candidate.xgboostScore} color="bg-blue-500" />
        <ScoreBar label="Neural Net" score={candidate.neuralNetScore} color="bg-purple-500" />
        <ScoreBar label="Ensemble" score={candidate.ensembleScore} color="bg-primary" />

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
          {candidate.electronPhononCoupling != null && (
            <div className="p-2 bg-blue-50 dark:bg-blue-950/30 rounded text-xs">
              <span className="text-muted-foreground block text-[10px]">e-ph coupling (lambda)</span>
              <span className="font-mono font-bold">{safeDisplay(candidate.electronPhononCoupling, 3)}</span>
            </div>
          )}
          {candidate.correlationStrength != null && (
            <div className="p-2 bg-purple-50 dark:bg-purple-950/30 rounded text-xs">
              <span className="text-muted-foreground block text-[10px]">Correlation U/W</span>
              <span className="font-mono font-bold">{safeDisplay(candidate.correlationStrength, 3)}</span>
            </div>
          )}
          {candidate.upperCriticalField != null && (
            <div className="p-2 bg-green-50 dark:bg-green-950/30 rounded text-xs">
              <span className="text-muted-foreground block text-[10px]">Upper Hc2</span>
              <span className="font-mono font-bold">{safeDisplay(candidate.upperCriticalField, 1)} T</span>
            </div>
          )}
          {candidate.coherenceLength != null && (
            <div className="p-2 bg-amber-50 dark:bg-amber-950/30 rounded text-xs">
              <span className="text-muted-foreground block text-[10px]">Coherence Length</span>
              <span className="font-mono font-bold">{safeDisplay(candidate.coherenceLength, 1)} nm</span>
            </div>
          )}
          {candidate.londonPenetrationDepth != null && (
            <div className="p-2 bg-teal-50 dark:bg-teal-950/30 rounded text-xs">
              <span className="text-muted-foreground block text-[10px]">London depth</span>
              <span className="font-mono font-bold">{safeDisplay(candidate.londonPenetrationDepth, 0)} nm</span>
            </div>
          )}
          {candidate.criticalCurrentDensity != null && (
            <div className="p-2 bg-orange-50 dark:bg-orange-950/30 rounded text-xs">
              <span className="text-muted-foreground block text-[10px]">Jc</span>
              <span className="font-mono font-bold">{safeDisplay(candidate.criticalCurrentDensity, 0)} A/cm2</span>
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-1 mt-2">
          {candidate.pairingMechanism && <Badge variant="outline" className="text-[10px]">{candidate.pairingMechanism}</Badge>}
          {candidate.pairingSymmetry && <Badge variant="outline" className="text-[10px]">{candidate.pairingSymmetry}</Badge>}
          {candidate.dimensionality && <Badge variant="outline" className="text-[10px]">{candidate.dimensionality}</Badge>}
          {candidate.fermiSurfaceTopology && <Badge variant="outline" className="text-[10px]">{candidate.fermiSurfaceTopology}</Badge>}
        </div>

        {candidate.cooperPairMechanism && (
          <p className="text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">Cooper Pair Description: </span>
            {candidate.cooperPairMechanism}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function CrystalStructuresSection({ structures }: { structures: CrystalStructure[] }) {
  if (structures.length === 0) return null;
  return (
    <Card data-testid="crystal-structures">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Layers className="h-5 w-5" />
          Crystal Structures ({structures.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {structures.map(s => {
            const lattice = s.latticeParams as Record<string, number> | null;
            return (
              <div key={s.id} className="p-3 bg-muted/50 rounded-md space-y-2" data-testid={`crystal-detail-${s.id}`}>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-mono text-sm font-bold">{s.spaceGroup}</p>
                    <p className="text-xs text-muted-foreground">{s.crystalSystem}</p>
                  </div>
                  <div className="flex gap-1">
                    {s.isStable && <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">Stable</Badge>}
                    {s.isMetastable && <Badge className="bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300 border-0 text-[10px]">Metastable</Badge>}
                    <Badge variant="outline" className="text-[10px]">{s.dimensionality}</Badge>
                  </div>
                </div>
                {s.prototype && <p className="text-xs text-muted-foreground"><span className="font-semibold text-foreground">Prototype: </span>{s.prototype}</p>}
                {lattice && (
                  <div className="grid grid-cols-3 gap-1 text-[10px]">
                    <div className="p-1 bg-background rounded text-center">
                      <span className="text-muted-foreground">a=</span><span className="font-mono font-bold">{Number.isFinite(lattice.a) ? lattice.a.toFixed(2) : "--"}</span>
                    </div>
                    <div className="p-1 bg-background rounded text-center">
                      <span className="text-muted-foreground">b=</span><span className="font-mono font-bold">{Number.isFinite(lattice.b) ? lattice.b.toFixed(2) : "--"}</span>
                    </div>
                    <div className="p-1 bg-background rounded text-center">
                      <span className="text-muted-foreground">c=</span><span className="font-mono font-bold">{Number.isFinite(lattice.c) ? lattice.c.toFixed(2) : "--"}</span>
                    </div>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {s.convexHullDistance != null && Number.isFinite(s.convexHullDistance) && (
                    <div><span className="text-muted-foreground">Hull dist: </span><span className="font-mono">{s.convexHullDistance.toFixed(3)} eV/atom</span></div>
                  )}
                  {s.synthesizability != null && Number.isFinite(s.synthesizability) && (
                    <div><span className="text-muted-foreground">Synth: </span><span className="font-mono">{(s.synthesizability * 100).toFixed(0)}%</span></div>
                  )}
                </div>
                {s.synthesisNotes && <p className="text-xs text-muted-foreground">{s.synthesisNotes}</p>}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function SynthesisSection({ processes }: { processes: SynthesisProcess[] }) {
  if (processes.length === 0) return null;
  return (
    <Card data-testid="synthesis-processes">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <FlaskConical className="h-5 w-5" />
          Synthesis Pathways ({processes.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {processes.map(proc => {
          const conditions = proc.conditions as Record<string, any> | null;
          return (
            <div key={proc.id} className="p-3 bg-muted/50 rounded-md space-y-2" data-testid={`synthesis-${proc.id}`}>
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-sm font-bold">{proc.method}</p>
                  <p className="text-xs text-muted-foreground">{proc.materialName}</p>
                </div>
                <Badge variant="outline" className="text-[10px]">{proc.difficulty}</Badge>
              </div>
              {proc.steps && proc.steps.length > 0 && (
                <ol className="text-xs space-y-1 ml-4 list-decimal text-muted-foreground">
                  {proc.steps.map((step, i) => <li key={i}>{step}</li>)}
                </ol>
              )}
              <div className="flex flex-wrap gap-2 text-xs">
                {proc.precursors && proc.precursors.length > 0 && (
                  <div>
                    <span className="text-muted-foreground">Precursors: </span>
                    <span className="font-mono">{proc.precursors.join(", ")}</span>
                  </div>
                )}
                {proc.timeEstimate && (
                  <div>
                    <span className="text-muted-foreground">Time: </span>
                    <span className="font-mono">{proc.timeEstimate}</span>
                  </div>
                )}
                {proc.yieldPercent != null && (
                  <div>
                    <span className="text-muted-foreground">Yield: </span>
                    <span className="font-mono">{proc.yieldPercent}%</span>
                  </div>
                )}
              </div>
              {conditions && (
                <div className="flex flex-wrap gap-2 text-[10px]">
                  {conditions.temperature && <Badge variant="secondary">{conditions.temperature}</Badge>}
                  {conditions.pressure && <Badge variant="secondary">{conditions.pressure}</Badge>}
                  {conditions.atmosphere && <Badge variant="secondary">{conditions.atmosphere}</Badge>}
                </div>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function ReactionsSection({ reactions }: { reactions: ChemicalReaction[] }) {
  if (reactions.length === 0) return null;
  return (
    <Card data-testid="chemical-reactions">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Beaker className="h-5 w-5" />
          Chemical Reactions ({reactions.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {reactions.map(rxn => {
          const energetics = rxn.energetics as Record<string, any> | null;
          return (
            <div key={rxn.id} className="p-3 bg-muted/50 rounded-md space-y-2" data-testid={`reaction-${rxn.id}`}>
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm font-bold">{rxn.name}</p>
                <Badge variant="outline" className="text-[10px]">{rxn.reactionType}</Badge>
              </div>
              <p className="font-mono text-xs bg-background p-2 rounded">{rxn.equation}</p>
              {energetics && (
                <div className="flex flex-wrap gap-3 text-xs">
                  {energetics.deltaH != null && (
                    <div><span className="text-muted-foreground">deltaH: </span><span className="font-mono">{energetics.deltaH} kJ/mol</span></div>
                  )}
                  {energetics.deltaG != null && (
                    <div><span className="text-muted-foreground">deltaG: </span><span className="font-mono">{energetics.deltaG} kJ/mol</span></div>
                  )}
                  {energetics.activationEnergy != null && (
                    <div><span className="text-muted-foreground">Ea: </span><span className="font-mono">{energetics.activationEnergy} kJ/mol</span></div>
                  )}
                </div>
              )}
              {rxn.mechanism && <p className="text-xs text-muted-foreground">{rxn.mechanism}</p>}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

interface CrossValidationEntry {
  source: string;
  property: string;
  predictedValue: number | null;
  externalValue: number;
  deviationPercent: number | null;
  agreement: "match" | "minor-discrepancy" | "major-discrepancy" | "no-comparison";
  unit: string;
}

interface CrossValidationData {
  formula: string;
  materialsProject: {
    available: boolean;
    hasData: boolean;
    summary: any;
    elasticity: any;
    phonon: any;
    magnetism: any;
  };
  aflow: {
    available: boolean;
    hasData: boolean;
    entries: any[];
  };
  crossValidation: CrossValidationEntry[];
  hasDiscrepancies: boolean;
}

const AGREEMENT_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  "match": { bg: "bg-green-100 dark:bg-green-950", text: "text-green-700 dark:text-green-400", label: "Match" },
  "minor-discrepancy": { bg: "bg-yellow-100 dark:bg-yellow-950", text: "text-yellow-700 dark:text-yellow-400", label: "Minor Deviation" },
  "major-discrepancy": { bg: "bg-red-100 dark:bg-red-950", text: "text-red-700 dark:text-red-400", label: "Major Deviation (>30%)" },
  "no-comparison": { bg: "bg-gray-100 dark:bg-gray-900", text: "text-gray-600 dark:text-gray-400", label: "Reference Only" },
};

function ExternalDataSourcesSection({ formula }: { formula: string }) {
  const { data, isLoading } = useQuery<CrossValidationData>({
    queryKey: ["/api/cross-validation", encodeURIComponent(formula)],
    enabled: !!formula,
  });

  if (isLoading) {
    return (
      <Card data-testid="external-data-sources">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Database className="h-5 w-5" />
            External Data Sources
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-32" />
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const mpStatus = data.materialsProject;
  const aflowStatus = data.aflow;
  const validations = data.crossValidation ?? [];

  return (
    <Card data-testid="external-data-sources">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Database className="h-5 w-5" />
          External Data Sources
          {data.hasDiscrepancies && (
            <Badge className="bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400 border-0 text-[10px]">
              <AlertTriangle className="h-3 w-3 mr-1" />
              Discrepancies Found
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="p-3 bg-muted/50 rounded-md" data-testid="mp-status">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-sm font-bold">Materials Project</span>
              {mpStatus.available ? (
                mpStatus.hasData ? (
                  <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">Data Found</Badge>
                ) : (
                  <Badge variant="outline" className="text-[10px]">No Match</Badge>
                )
              ) : (
                <Badge variant="outline" className="text-[10px]">API Key Required</Badge>
              )}
            </div>
            {mpStatus.hasData && mpStatus.summary && (
              <div className="space-y-1 text-xs">
                <div className="flex justify-between gap-1">
                  <span className="text-muted-foreground">E above hull</span>
                  <span className="font-mono">{mpStatus.summary.energyAboveHull?.toFixed(4)} eV/atom</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span className="text-muted-foreground">Band Gap</span>
                  <span className="font-mono">{mpStatus.summary.bandGap?.toFixed(3)} eV</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span className="text-muted-foreground">Metallic</span>
                  <span className="font-mono">{mpStatus.summary.isMetallic ? "Yes" : "No"}</span>
                </div>
                {mpStatus.summary.spaceGroup && (
                  <div className="flex justify-between gap-1">
                    <span className="text-muted-foreground">Space Group</span>
                    <span className="font-mono">{mpStatus.summary.spaceGroup}</span>
                  </div>
                )}
              </div>
            )}
            {mpStatus.hasData && mpStatus.elasticity && (
              <div className="space-y-1 text-xs mt-2 pt-2 border-t border-border/50">
                <div className="flex justify-between gap-1">
                  <span className="text-muted-foreground">Bulk Modulus</span>
                  <span className="font-mono">{mpStatus.elasticity.bulkModulus?.toFixed(1)} GPa</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span className="text-muted-foreground">Shear Modulus</span>
                  <span className="font-mono">{mpStatus.elasticity.shearModulus?.toFixed(1)} GPa</span>
                </div>
              </div>
            )}
            {mpStatus.hasData && mpStatus.phonon && mpStatus.phonon.hasPhononData && (
              <div className="space-y-1 text-xs mt-2 pt-2 border-t border-border/50">
                <div className="flex justify-between gap-1">
                  <span className="text-muted-foreground">Phonon Data</span>
                  <span className="font-mono">Available</span>
                </div>
                {mpStatus.phonon.lastPhononFreq != null && (
                  <div className="flex justify-between gap-1">
                    <span className="text-muted-foreground">Last Phonon Freq</span>
                    <span className="font-mono">{mpStatus.phonon.lastPhononFreq.toFixed(1)} THz</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="p-3 bg-muted/50 rounded-md" data-testid="aflow-status">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-sm font-bold">AFLOW</span>
              {aflowStatus.hasData ? (
                <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">
                  {aflowStatus.entries.length} {aflowStatus.entries.length === 1 ? "Entry" : "Entries"}
                </Badge>
              ) : (
                <Badge variant="outline" className="text-[10px]">No Match</Badge>
              )}
            </div>
            {aflowStatus.hasData && aflowStatus.entries[0] && (
              <div className="space-y-1 text-xs">
                {aflowStatus.entries[0].spaceGroupSymbol && (
                  <div className="flex justify-between gap-1">
                    <span className="text-muted-foreground">Space Group</span>
                    <span className="font-mono">{aflowStatus.entries[0].spaceGroupSymbol}</span>
                  </div>
                )}
                {aflowStatus.entries[0].enthalpy_formation_atom != null && (
                  <div className="flex justify-between gap-1">
                    <span className="text-muted-foreground">Formation H</span>
                    <span className="font-mono">{aflowStatus.entries[0].enthalpy_formation_atom.toFixed(4)} eV/atom</span>
                  </div>
                )}
                {aflowStatus.entries[0].bandgap != null && (
                  <div className="flex justify-between gap-1">
                    <span className="text-muted-foreground">Band Gap</span>
                    <span className="font-mono">{aflowStatus.entries[0].bandgap.toFixed(3)} eV</span>
                  </div>
                )}
                {aflowStatus.entries[0].Bvoigt != null && (
                  <div className="flex justify-between gap-1">
                    <span className="text-muted-foreground">Bulk Modulus</span>
                    <span className="font-mono">{aflowStatus.entries[0].Bvoigt.toFixed(1)} GPa</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {validations.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-semibold text-muted-foreground">Cross-Validation Results</p>
            <div className="space-y-1.5">
              {validations.map((v: CrossValidationEntry, idx: number) => {
                const style = AGREEMENT_STYLES[v.agreement] ?? AGREEMENT_STYLES["no-comparison"];
                return (
                  <div
                    key={idx}
                    className="flex items-center justify-between gap-2 p-2 bg-muted/30 rounded text-xs"
                    data-testid={`cross-validation-${idx}`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <Badge variant="outline" className="text-[10px] shrink-0">{v.source}</Badge>
                      <span className="text-muted-foreground truncate">{v.property}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="font-mono">
                        {typeof v.externalValue === "number" ? v.externalValue.toFixed(3) : "--"} {v.unit}
                      </span>
                      <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${style.bg} ${style.text}`}>
                        {v.agreement === "major-discrepancy" && <AlertTriangle className="h-2.5 w-2.5 mr-0.5" />}
                        {v.agreement === "match" && <CheckCircle2 className="h-2.5 w-2.5 mr-0.5" />}
                        {style.label}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {validations.length === 0 && !mpStatus.hasData && !aflowStatus.hasData && (
          <p className="text-xs text-muted-foreground text-center py-2">
            No external data found for this composition in Materials Project or AFLOW databases.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function PipelineResultsSection({ results }: { results: ComputationalResult[] }) {
  if (results.length === 0) return null;

  const STAGE_NAMES = ["ML Filter", "Electronic Structure", "Phonon / E-Ph Coupling", "Tc Prediction (Eliashberg)", "Synthesis Feasibility"];

  return (
    <Card data-testid="pipeline-results">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Zap className="h-5 w-5" />
          Pipeline Results ({results.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {results.map(r => {
          const resultData = r.results as Record<string, any>;
          return (
            <div key={r.id} className="p-3 bg-muted/50 rounded-md space-y-1.5" data-testid={`pipeline-result-${r.id}`}>
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-sm font-bold">{STAGE_NAMES[r.pipelineStage] ?? `Stage ${r.pipelineStage}`}</p>
                  <p className="text-xs text-muted-foreground">{r.computationType}</p>
                </div>
                <div className="flex gap-1">
                  {r.passed ? (
                    <Badge className="bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-0 text-[10px]">Passed</Badge>
                  ) : (
                    <Badge variant="destructive" className="text-[10px]">Failed</Badge>
                  )}
                  {r.confidence != null && (
                    <Badge variant="outline" className="text-[10px]">{(safeNum(r.confidence) * 100).toFixed(0)}% conf</Badge>
                  )}
                </div>
              </div>
              {r.failureReason && (
                <div className="flex items-start gap-2 text-xs p-2 bg-red-50 dark:bg-red-950/20 rounded">
                  <XCircle className="h-3.5 w-3.5 text-red-500 mt-0.5 shrink-0" />
                  <span className="text-red-700 dark:text-red-400">{r.failureReason}</span>
                </div>
              )}
              {r.computeTimeMs != null && (
                <p className="text-[10px] text-muted-foreground">
                  {r.computeTimeMs < 100
                    ? "Surrogate model (heuristic estimate)"
                    : `Compute time: ${r.computeTimeMs}ms`}
                </p>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function ExperimentalValidationSection({ formula }: { formula: string }) {
  const [showForm, setShowForm] = useState(false);
  const [validationType, setValidationType] = useState("");
  const [result, setResult] = useState("");
  const [measuredTc, setMeasuredTc] = useState("");
  const [measuredPressure, setMeasuredPressure] = useState("");
  const [notes, setNotes] = useState("");

  const { data: validations, isLoading } = useQuery<any[]>({
    queryKey: ["/api/validations", encodeURIComponent(formula)],
    enabled: !!formula,
  });

  const mutation = useMutation({
    mutationFn: async (body: any) => {
      await apiRequest("POST", "/api/validations", body);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/validations", encodeURIComponent(formula)] });
      setShowForm(false);
      setValidationType("");
      setResult("");
      setMeasuredTc("");
      setMeasuredPressure("");
      setNotes("");
    },
  });

  const handleSubmit = () => {
    if (!validationType || !result) return;
    mutation.mutate({
      formula,
      validationType,
      result,
      measuredTc: measuredTc ? parseFloat(measuredTc) : null,
      measuredPressure: measuredPressure ? parseFloat(measuredPressure) : null,
      notes: notes || null,
    });
  };

  const RESULT_COLORS: Record<string, string> = {
    confirmed: "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
    partial: "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
    failed: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
    inconclusive: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };

  return (
    <Card data-testid="experimental-validations">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <ClipboardCheck className="h-5 w-5" />
            Experimental Validations ({validations?.length ?? 0})
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowForm(!showForm)}
            data-testid="button-log-validation"
          >
            <Plus className="h-3.5 w-3.5 mr-1" />
            Log Validation
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {showForm && (
          <div className="p-3 bg-muted/50 rounded-md space-y-3 border border-border" data-testid="validation-form">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Test Type</label>
                <Select value={validationType} onValueChange={setValidationType}>
                  <SelectTrigger data-testid="select-validation-type"><SelectValue placeholder="Select type" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="resistance">Resistance Measurement</SelectItem>
                    <SelectItem value="meissner">Meissner Effect Test</SelectItem>
                    <SelectItem value="xrd">X-Ray Diffraction</SelectItem>
                    <SelectItem value="tc_measurement">Tc Measurement</SelectItem>
                    <SelectItem value="pressure_test">Pressure Test</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Result</label>
                <Select value={result} onValueChange={setResult}>
                  <SelectTrigger data-testid="select-validation-result"><SelectValue placeholder="Select result" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="confirmed">Confirmed</SelectItem>
                    <SelectItem value="partial">Partial</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                    <SelectItem value="inconclusive">Inconclusive</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Measured Tc (K)</label>
                <Input
                  type="number"
                  placeholder="Optional"
                  value={measuredTc}
                  onChange={e => setMeasuredTc(e.target.value)}
                  data-testid="input-measured-tc"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Pressure (GPa)</label>
                <Input
                  type="number"
                  placeholder="Optional"
                  value={measuredPressure}
                  onChange={e => setMeasuredPressure(e.target.value)}
                  data-testid="input-measured-pressure"
                />
              </div>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Notes</label>
              <Textarea
                placeholder="Lab notes..."
                value={notes}
                onChange={e => setNotes(e.target.value)}
                className="h-16"
                data-testid="textarea-validation-notes"
              />
            </div>
            <div className="flex gap-2 justify-end">
              <Button variant="outline" size="sm" onClick={() => setShowForm(false)} data-testid="button-cancel-validation">Cancel</Button>
              <Button size="sm" onClick={handleSubmit} disabled={!validationType || !result || mutation.isPending} data-testid="button-submit-validation">
                {mutation.isPending ? "Saving..." : "Save"}
              </Button>
            </div>
          </div>
        )}

        {isLoading ? (
          <Skeleton className="h-20" />
        ) : validations && validations.length > 0 ? (
          <div className="space-y-2">
            {validations.map((v: any) => (
              <div key={v.id} className="p-3 bg-muted/50 rounded-md" data-testid={`validation-entry-${v.id}`}>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium capitalize">{v.validationType?.replace(/_/g, " ")}</p>
                    <p className="text-[10px] text-muted-foreground">
                      {v.performedAt ? new Date(v.performedAt).toLocaleDateString() : "Unknown date"}
                    </p>
                  </div>
                  <Badge className={`${RESULT_COLORS[v.result] ?? RESULT_COLORS.inconclusive} border-0 text-[10px]`}>
                    {v.result}
                  </Badge>
                </div>
                {(v.measuredTc != null || v.measuredPressure != null) && (
                  <div className="flex gap-4 mt-1.5 text-xs">
                    {v.measuredTc != null && <span className="text-muted-foreground">Tc: <span className="font-mono font-medium text-foreground">{v.measuredTc}K</span></span>}
                    {v.measuredPressure != null && <span className="text-muted-foreground">P: <span className="font-mono font-medium text-foreground">{v.measuredPressure} GPa</span></span>}
                  </div>
                )}
                {v.notes && <p className="text-xs text-muted-foreground mt-1.5">{v.notes}</p>}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-4">No experimental validations logged yet</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function CandidateDetail() {
  const [, params] = useRoute("/candidate/:formula");
  const formula = params?.formula ? decodeURIComponent(params.formula) : "";

  const { data, isLoading } = useQuery<CandidateProfile>({
    queryKey: ["/api/candidate-profile", encodeURIComponent(formula)],
    enabled: !!formula,
  });

  const { data: calibrationData } = useQuery<CalibrationResponse>({
    queryKey: ["/api/ml-calibration"],
  });

  const p90 = calibrationData?.absResidualPercentiles?.p90;

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-48" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </div>
    );
  }

  const candidate = data?.candidates?.[0];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/superconductor" data-testid="link-back-sc-lab">
          <button className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="h-4 w-4" />
            Back to SC Lab
          </button>
        </Link>
      </div>

      <div>
        <h1 className="text-2xl font-bold tracking-tight" data-testid="text-candidate-formula">
          {formula}
        </h1>
        <p className="text-muted-foreground mt-1">
          Unified candidate profile across all analysis pipelines
        </p>
      </div>

      {!candidate && !isLoading && (
        <Card>
          <CardContent className="pt-8 pb-8 text-center text-muted-foreground">
            <Atom className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No superconductor candidate data found for {formula}</p>
            <p className="text-xs mt-1">Data may still be accumulating from the research pipeline</p>
          </CardContent>
        </Card>
      )}

      {candidate && <CandidateHeader candidate={candidate} p90={p90} />}

      {formula && <ExternalDataSourcesSection formula={formula} />}

      {formula && <ExperimentalValidationSection formula={formula} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {candidate && <MLScoresSection candidate={candidate} />}
        <CrystalStructuresSection structures={data?.crystalStructures ?? []} />
        <SynthesisSection processes={data?.synthesisProcesses ?? []} />
        <ReactionsSection reactions={data?.chemicalReactions ?? []} />
        <PipelineResultsSection results={data?.computationalResults ?? []} />
      </div>
    </div>
  );
}
