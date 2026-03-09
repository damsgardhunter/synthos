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

        {candidate.notes && (
          <div className="mt-2 space-y-1">
            {candidate.notes.includes("[Inverse design:") && (
              <Badge className="bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300 border-0 text-[10px]" data-testid="badge-inverse-design">
                Inverse Design
              </Badge>
            )}
            {candidate.notes.includes("[Structural variant:") && (
              <Badge className="bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300 border-0 text-[10px]" data-testid="badge-structural-variant">
                Structural Variant
              </Badge>
            )}
            {(() => {
              const noveltyMatch = candidate.notes.match(/novelty=([\d.]+)/);
              if (noveltyMatch) {
                const novelty = parseFloat(noveltyMatch[1]);
                return (
                  <div className="text-[10px] text-muted-foreground" data-testid="text-structural-novelty">
                    Structural Novelty: <span className="font-mono font-bold">{novelty.toFixed(2)}</span>
                  </div>
                );
              }
              return null;
            })()}
            {(() => {
              const psMatch = candidate.notes.match(/PS=([\d.]+)/);
              if (psMatch) {
                const ps = parseFloat(psMatch[1]);
                return (
                  <div className="text-[10px] text-muted-foreground" data-testid="text-pairing-susceptibility">
                    Pairing Susceptibility: <span className="font-mono font-bold">{ps.toFixed(3)}</span>
                  </div>
                );
              }
              return null;
            })()}
            {(() => {
              const instMatch = candidate.notes.match(/\[Instability: (.+?)=([\d.]+), QCP=([\d.]+), CDW=([\d.]+), MIT=([\d.]+)\]/);
              if (instMatch) {
                const boundary = instMatch[1];
                const overall = parseFloat(instMatch[2]);
                const qcp = parseFloat(instMatch[3]);
                const cdw = parseFloat(instMatch[4]);
                const mit = parseFloat(instMatch[5]);
                return (
                  <div className="mt-1.5 p-2 bg-muted/50 rounded space-y-1" data-testid="instability-proximity">
                    <div className="text-[10px] font-semibold text-foreground">Instability Proximity</div>
                    <div className="text-[10px] text-muted-foreground">
                      Nearest: <span className="font-mono font-bold">{boundary}</span> ({overall.toFixed(2)})
                    </div>
                    <div className="flex gap-3 text-[10px] text-muted-foreground">
                      <span>QCP: <span className="font-mono">{qcp.toFixed(2)}</span></span>
                      <span>CDW: <span className="font-mono">{cdw.toFixed(2)}</span></span>
                      <span>MIT: <span className="font-mono">{mit.toFixed(2)}</span></span>
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${overall > 0.6 ? "bg-red-500" : overall > 0.3 ? "bg-yellow-500" : "bg-blue-500"}`}
                        style={{ width: `${Math.min(100, overall * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              }
              return null;
            })()}
            {(() => {
              const pairingMatch = candidate.notes.match(/\[Pairing: (.+?) \(Tc=([\d.]+)K, conf=([\d.]+)\)\]/);
              if (pairingMatch) {
                const mechanism = pairingMatch[1];
                const tc = parseFloat(pairingMatch[2]);
                const conf = parseFloat(pairingMatch[3]);
                return (
                  <div className="mt-1 text-[10px] text-muted-foreground" data-testid="text-pairing-analysis">
                    Dominant Pairing: <span className="font-mono font-bold">{mechanism}</span> (Tc={tc.toFixed(0)}K, confidence={conf.toFixed(2)})
                  </div>
                );
              }
              return null;
            })()}
          </div>
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

interface NovelSynthesisRoute {
  method: string;
  steps: string[];
  temperatureProfile: string;
  pressureProfile: string;
  estimatedCoolingRate: string;
  atmosphere: string;
  expectedYieldRange: string;
  noveltyScore: number;
  synthesisConfidence: "high" | "medium" | "low";
  physicsJustification: string;
  source: "physics-reasoned" | "literature-based";
  keyInnovation: string;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  medium: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
  low: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
};

function NovelSynthesisSection({ candidate }: { candidate: SuperconductorCandidate }) {
  const synthPath = candidate.synthesisPath as { routes?: any[]; lastUpdated?: string } | null;
  const allRoutes = synthPath?.routes ?? [];
  const physicsRoutes: NovelSynthesisRoute[] = allRoutes.filter(
    (r: any) => r.source === "physics-reasoned" && r.steps
  );
  const plannerRoutes = allRoutes.filter(
    (r: any) => r.source === "synthesis-planner"
  );
  const heuristicRoutes = allRoutes.filter(
    (r: any) => r.source === "heuristic-generator"
  );
  const analogyRoutes = allRoutes.filter(
    (r: any) => r.source === "analogy-transfer"
  );
  const literatureRoutes = allRoutes.filter(
    (r: any) => !["physics-reasoned", "synthesis-planner", "heuristic-generator", "analogy-transfer", "reaction-pathway-engine"].includes(r.source)
  );
  const pathwayRoutes = allRoutes.filter(
    (r: any) => r.source === "reaction-pathway-engine"
  );

  const totalRoutes = allRoutes.length;
  if (totalRoutes === 0) return null;

  return (
    <Card data-testid="novel-synthesis-routes" className="lg:col-span-2">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Synthesis Routes ({totalRoutes})
          </CardTitle>
          {synthPath?.lastUpdated && (
            <span className="text-[10px] text-muted-foreground">
              Updated {new Date(synthPath.lastUpdated).toLocaleDateString()}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {plannerRoutes.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300 border-0 text-[10px]" data-testid="badge-planner">
                Planned Routes
              </Badge>
              <span className="text-xs text-muted-foreground">{plannerRoutes.length} route(s) from synthesis planner</span>
            </div>
            {plannerRoutes.map((route: any, i: number) => (
              <div key={`planner-${i}`} className="p-3 bg-blue-50/50 dark:bg-blue-950/20 rounded-md space-y-2 border border-blue-200/50 dark:border-blue-800/30" data-testid={`synthesis-planner-${i}`}>
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-bold">{route.routeName || route.method || "Planned route"}</p>
                  {typeof route.feasibilityScore === "number" && (
                    <Badge variant="outline" className="text-[10px] font-mono shrink-0" data-testid={`planner-feas-${i}`}>
                      {(route.feasibilityScore * 100).toFixed(1)}% feasible
                    </Badge>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 text-[10px] text-muted-foreground">
                  {route.method && <span>Method: {route.method}</span>}
                  {route.difficulty && <span>Difficulty: {route.difficulty}</span>}
                  {route.maxTemperature != null && <span>Max T: {route.maxTemperature}K</span>}
                  {route.maxPressure != null && <span>Max P: {route.maxPressure} GPa</span>}
                </div>
                {route.precursors && Array.isArray(route.precursors) && route.precursors.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {route.precursors.map((p: string, j: number) => (
                      <Badge key={j} variant="secondary" className="text-[10px] font-mono border-0">{p}</Badge>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {heuristicRoutes.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 border-0 text-[10px]" data-testid="badge-heuristic">
                Rule-Based
              </Badge>
              <span className="text-xs text-muted-foreground">{heuristicRoutes.length} route(s) from heuristic rules</span>
            </div>
            {heuristicRoutes.map((route: any, i: number) => (
              <div key={`heuristic-${i}`} className="p-3 bg-emerald-50/50 dark:bg-emerald-950/20 rounded-md space-y-2 border border-emerald-200/50 dark:border-emerald-800/30" data-testid={`synthesis-heuristic-${i}`}>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-bold">{route.routeName || route.method}</p>
                    {route.equation && (
                      <p className="text-xs font-mono text-muted-foreground mt-0.5">{route.equation}</p>
                    )}
                  </div>
                  {typeof route.confidence === "number" && (
                    <Badge variant="outline" className="text-[10px] font-mono shrink-0" data-testid={`heuristic-conf-${i}`}>
                      {(route.confidence * 100).toFixed(0)}% confidence
                    </Badge>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 text-[10px] text-muted-foreground">
                  {route.rule && <span>Rule: {route.rule}</span>}
                  {route.difficulty && <span>Difficulty: {route.difficulty}</span>}
                  {route.temperature != null && <span>{route.temperature}K</span>}
                  {route.pressure != null && route.pressure > 0 && <span>{route.pressure} GPa</span>}
                  {route.atmosphere && <span>{route.atmosphere}</span>}
                </div>
                {route.precursors && Array.isArray(route.precursors) && route.precursors.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {route.precursors.map((p: string, j: number) => (
                      <Badge key={j} variant="secondary" className="text-[10px] font-mono border-0">{p}</Badge>
                    ))}
                  </div>
                )}
                {route.steps && Array.isArray(route.steps) && (
                  <ol className="text-xs space-y-0.5 ml-4 list-decimal text-muted-foreground">
                    {route.steps.slice(0, 6).map((s: string, j: number) => <li key={j}>{s}</li>)}
                    {route.steps.length > 6 && <li className="italic">...and {route.steps.length - 6} more step(s)</li>}
                  </ol>
                )}
                {route.notes && (
                  <p className="text-[10px] text-muted-foreground italic">{route.notes}</p>
                )}
              </div>
            ))}
          </div>
        )}
        {physicsRoutes.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-primary/10 text-primary border border-primary/30 text-[10px]" data-testid="badge-physics-reasoned">
                Physics-Reasoned
              </Badge>
              <span className="text-xs text-muted-foreground">{physicsRoutes.length} route(s) derived from first-principles analysis</span>
            </div>
            {physicsRoutes.map((route, i) => (
              <NovelRouteCard key={`novel-${i}`} route={route} index={i} />
            ))}
          </div>
        )}
        {analogyRoutes.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300 border-0 text-[10px]" data-testid="badge-analogy">
                Analogy Transfer
              </Badge>
              <span className="text-xs text-muted-foreground">{analogyRoutes.length} route(s) from similar materials</span>
            </div>
            {analogyRoutes.map((route: any, i: number) => (
              <div key={`analogy-${i}`} className="p-3 bg-purple-50/30 dark:bg-purple-950/20 rounded-md space-y-1" data-testid={`synthesis-analogy-${i}`}>
                <p className="text-sm font-bold">{route.method || route.routeName || "Analogy route"}</p>
                {route.steps && Array.isArray(route.steps) && (
                  <ol className="text-xs space-y-0.5 ml-4 list-decimal text-muted-foreground">
                    {route.steps.slice(0, 5).map((s: string, j: number) => <li key={j}>{typeof s === "string" ? s : JSON.stringify(s)}</li>)}
                  </ol>
                )}
              </div>
            ))}
          </div>
        )}
        {(pathwayRoutes.length > 0 || literatureRoutes.length > 0) && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-[10px]" data-testid="badge-literature-based">
                {pathwayRoutes.length > 0 ? "Pathway / Literature" : "Literature-Based"}
              </Badge>
              <span className="text-xs text-muted-foreground">{pathwayRoutes.length + literatureRoutes.length} route(s)</span>
            </div>
            {[...pathwayRoutes, ...literatureRoutes].map((route: any, i: number) => (
              <div key={`lit-${i}`} className="p-3 bg-muted/30 rounded-md space-y-1" data-testid={`synthesis-literature-${i}`}>
                <p className="text-sm font-bold">{route.method || route.routeName || route.name || "Unknown method"}</p>
                {route.steps && Array.isArray(route.steps) && (
                  <ol className="text-xs space-y-0.5 ml-4 list-decimal text-muted-foreground">
                    {route.steps.slice(0, 5).map((s: string, j: number) => <li key={j}>{typeof s === "string" ? s : JSON.stringify(s)}</li>)}
                  </ol>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RetrosynthesisSection({ formula }: { formula: string }) {
  const { data } = useQuery<{
    formula: string;
    family: string;
    totalRoutes: number;
    routes: Array<{
      type: string;
      precursors: string[];
      product: string;
      equation: string;
      deltaE: number;
      complexity: number;
      availability: number;
      similarity: number;
      overallScore: number;
      reasoning: string[];
    }>;
    bestRoute: any;
    analysisNotes: string[];
  }>({ queryKey: ["/api/retrosynthesis/routes", formula] });

  const { data: mlData } = useQuery<{
    feasibility: number;
    confidence: number;
    features: Record<string, number>;
    reasoning: string[];
  }>({ queryKey: ["/api/ml-synthesis/predict", formula] });

  const { data: scoreData } = useQuery<{
    formula: string;
    score: number;
    thermodynamicFeasibility: number;
    precursorAvailability: number;
    structuralSimilarity: number;
    reactionComplexity: number;
  }>({ queryKey: ["/api/ml-synthesis/score", formula] });

  const { data: compData } = useQuery<{
    formula: string;
    featureCount: number;
    featureNames: string[];
    features: Record<string, number>;
  }>({ queryKey: ["/api/xgboost/composition-features", formula] });

  const { data: xgbUncertaintyData } = useQuery<{
    formula: string;
    tcMean: number;
    tcStd: number;
    normalizedUncertainty: number;
    score: number;
    perModelPredictions: number[];
    acquisitionScore: number;
    reasoning: string[];
  }>({ queryKey: ["/api/xgboost/uncertainty", formula] });

  const { data: gnnPredData } = useQuery<{
    formula: string;
    prediction: {
      tc: number;
      formationEnergy: number;
      lambda: number;
      uncertainty: number;
      phononStability: boolean;
      confidence: number;
    };
    graphStats: {
      nodes: number;
      edges: number;
      threeBodyFeatures: number;
      elements: string[];
    };
    modelVersion: number;
  }>({ queryKey: ["/api/gnn/predict", formula] });

  if (!data && !mlData) return null;

  const typeColors: Record<string, string> = {
    "direct-elemental": "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/40 dark:text-cyan-300",
    "binary-intermediate": "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    "oxide-decomposition": "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
    "precursor-substitution": "bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300",
  };

  return (
    <Card data-testid="retrosynthesis-section" className="lg:col-span-2">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <FlaskConical className="h-5 w-5" />
          Retrosynthesis & ML Feasibility
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {(mlData || scoreData) && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {mlData && (
              <>
                <div className="p-2 bg-violet-50/50 dark:bg-violet-950/20 rounded-md border border-violet-200/50 dark:border-violet-800/30" data-testid="ml-feasibility-score">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">ML Feasibility</p>
                  <p className="text-lg font-mono font-bold">{(mlData.feasibility * 100).toFixed(1)}%</p>
                </div>
                <div className="p-2 bg-violet-50/50 dark:bg-violet-950/20 rounded-md border border-violet-200/50 dark:border-violet-800/30" data-testid="ml-confidence">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">ML Confidence</p>
                  <p className="text-lg font-mono font-bold">{(mlData.confidence * 100).toFixed(0)}%</p>
                </div>
              </>
            )}
            {scoreData && (
              <>
                <div className="p-2 bg-indigo-50/50 dark:bg-indigo-950/20 rounded-md border border-indigo-200/50 dark:border-indigo-800/30" data-testid="synthesis-score">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Synthesis Score</p>
                  <p className="text-lg font-mono font-bold">{(scoreData.score * 100).toFixed(1)}%</p>
                </div>
                <div className="p-2 bg-indigo-50/50 dark:bg-indigo-950/20 rounded-md border border-indigo-200/50 dark:border-indigo-800/30" data-testid="thermo-feasibility">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Thermo Feasibility</p>
                  <p className="text-lg font-mono font-bold">{(scoreData.thermodynamicFeasibility * 100).toFixed(0)}%</p>
                </div>
              </>
            )}
          </div>
        )}
        {scoreData && (
          <div className="grid grid-cols-4 gap-1">
            {[
              { label: "Thermo", value: scoreData.thermodynamicFeasibility, weight: "40%" },
              { label: "Precursors", value: scoreData.precursorAvailability, weight: "30%" },
              { label: "Similarity", value: scoreData.structuralSimilarity, weight: "20%" },
              { label: "Complexity", value: scoreData.reactionComplexity, weight: "10%" },
            ].map((item) => (
              <div key={item.label} className="text-center" data-testid={`score-breakdown-${item.label.toLowerCase()}`}>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden mb-1">
                  <div
                    className="h-full bg-primary rounded-full"
                    style={{ width: `${(item.value * 100)}%` }}
                  />
                </div>
                <p className="text-[9px] text-muted-foreground">{item.label} ({item.weight})</p>
                <p className="text-[10px] font-mono font-bold">{(item.value * 100).toFixed(0)}%</p>
              </div>
            ))}
          </div>
        )}
        {mlData && mlData.reasoning.length > 0 && (
          <div className="space-y-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">ML Reasoning</p>
            <ul className="text-xs text-muted-foreground space-y-0.5 ml-3 list-disc">
              {mlData.reasoning.slice(0, 5).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}
        {data && data.routes.length > 0 && (
          <div className="space-y-3">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Retrosynthesis Routes ({data.totalRoutes})
            </p>
            {data.routes.slice(0, 6).map((route, i) => (
              <div
                key={`retro-${i}`}
                className="p-3 bg-muted/30 rounded-md space-y-2 border border-border/50"
                data-testid={`retro-route-${i}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-mono font-bold">{route.equation}</p>
                    <Badge className={`${typeColors[route.type] ?? "bg-muted text-muted-foreground"} border-0 text-[10px] mt-1`}>
                      {route.type}
                    </Badge>
                  </div>
                  <Badge variant="outline" className="text-[10px] font-mono shrink-0" data-testid={`retro-score-${i}`}>
                    {(route.overallScore * 100).toFixed(0)}% score
                  </Badge>
                </div>
                <div className="flex flex-wrap gap-3 text-[10px] text-muted-foreground">
                  <span>dE: {route.deltaE.toFixed(2)} eV</span>
                  <span>Avail: {(route.availability * 100).toFixed(0)}%</span>
                  <span>Complexity: {route.complexity.toFixed(2)}</span>
                  <span>Similarity: {(route.similarity * 100).toFixed(0)}%</span>
                </div>
                {route.precursors.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {route.precursors.map((p, j) => (
                      <Badge key={j} variant="secondary" className="text-[10px] font-mono border-0">{p}</Badge>
                    ))}
                  </div>
                )}
                {route.reasoning.length > 0 && (
                  <p className="text-[10px] text-muted-foreground italic">{route.reasoning[0]}</p>
                )}
              </div>
            ))}
          </div>
        )}
        {data && data.analysisNotes.length > 0 && (
          <div className="space-y-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Analysis Notes</p>
            <ul className="text-xs text-muted-foreground space-y-0.5 ml-3 list-disc">
              {data.analysisNotes.map((note, i) => <li key={i}>{note}</li>)}
            </ul>
          </div>
        )}
        {xgbUncertaintyData && (
          <div className="space-y-2" data-testid="xgb-ensemble-uncertainty-section">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              XGBoost Ensemble Uncertainty ({xgbUncertaintyData.perModelPredictions.length} bootstrap models)
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div className="p-2 bg-blue-50/50 dark:bg-blue-950/20 rounded-md border border-blue-200/50 dark:border-blue-800/30" data-testid="xgb-tc-mean">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Tc Mean</p>
                <p className="text-lg font-mono font-bold">{xgbUncertaintyData.tcMean.toFixed(1)}K</p>
              </div>
              <div className="p-2 bg-blue-50/50 dark:bg-blue-950/20 rounded-md border border-blue-200/50 dark:border-blue-800/30" data-testid="xgb-tc-std">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Tc Std Dev</p>
                <p className="text-lg font-mono font-bold">+/- {xgbUncertaintyData.tcStd.toFixed(1)}K</p>
              </div>
              <div className="p-2 bg-blue-50/50 dark:bg-blue-950/20 rounded-md border border-blue-200/50 dark:border-blue-800/30" data-testid="xgb-uncertainty">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Uncertainty</p>
                <p className={`text-lg font-mono font-bold ${xgbUncertaintyData.normalizedUncertainty > 0.6 ? "text-red-600 dark:text-red-400" : xgbUncertaintyData.normalizedUncertainty > 0.3 ? "text-amber-600 dark:text-amber-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                  {(xgbUncertaintyData.normalizedUncertainty * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-2 bg-blue-50/50 dark:bg-blue-950/20 rounded-md border border-blue-200/50 dark:border-blue-800/30" data-testid="xgb-acquisition">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Acquisition</p>
                <p className="text-lg font-mono font-bold">{xgbUncertaintyData.acquisitionScore.toFixed(3)}</p>
              </div>
            </div>
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-[10px] text-muted-foreground">Per-model Tc:</span>
              {xgbUncertaintyData.perModelPredictions.map((pred, i) => (
                <span key={i} className="text-[10px] font-mono px-1.5 py-0.5 bg-muted/60 rounded" data-testid={`xgb-model-pred-${i}`}>
                  M{i + 1}: {pred.toFixed(1)}K
                </span>
              ))}
            </div>
            {xgbUncertaintyData.reasoning.length > 0 && (
              <ul className="text-xs text-muted-foreground space-y-0.5 ml-3 list-disc">
                {xgbUncertaintyData.reasoning.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            )}
          </div>
        )}
        {gnnPredData && (
          <div className="space-y-2" data-testid="gnn-prediction-section">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              GNN Crystal Graph Prediction (v{gnnPredData.modelVersion})
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-tc">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">GNN Tc</p>
                <p className="text-lg font-mono font-bold">{gnnPredData.prediction.tc.toFixed(1)}K</p>
              </div>
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-uncertainty">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Uncertainty</p>
                <p className={`text-lg font-mono font-bold ${gnnPredData.prediction.uncertainty > 0.6 ? "text-red-600 dark:text-red-400" : gnnPredData.prediction.uncertainty > 0.3 ? "text-amber-600 dark:text-amber-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                  {(gnnPredData.prediction.uncertainty * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-lambda">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Lambda (e-ph)</p>
                <p className="text-lg font-mono font-bold">{gnnPredData.prediction.lambda.toFixed(3)}</p>
              </div>
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-confidence">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Confidence</p>
                <p className="text-lg font-mono font-bold">{(gnnPredData.prediction.confidence * 100).toFixed(0)}%</p>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="gnn-formation-energy">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Formation Energy</p>
                <p className="text-sm font-mono">{gnnPredData.prediction.formationEnergy.toFixed(3)} eV/atom</p>
              </div>
              <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="gnn-phonon-stability">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Phonon Stability</p>
                <p className={`text-sm font-mono ${gnnPredData.prediction.phononStability ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                  {gnnPredData.prediction.phononStability ? "Stable" : "Unstable"}
                </p>
              </div>
              <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="gnn-graph-size">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Crystal Graph</p>
                <p className="text-sm font-mono">{gnnPredData.graphStats.nodes}N / {gnnPredData.graphStats.edges}E / {gnnPredData.graphStats.threeBodyFeatures}T</p>
              </div>
              <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="gnn-elements">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Elements</p>
                <p className="text-sm font-mono">{gnnPredData.graphStats.elements.join(", ")}</p>
              </div>
            </div>
          </div>
        )}
        {compData && (
          <div className="space-y-2" data-testid="composition-features-section">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              XGBoost Composition Features ({compData.featureCount} features)
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1.5">
              {Object.entries(compData.features).filter(([, v]) => typeof v === "number" && v !== 0).slice(0, 20).map(([key, value]) => (
                <div key={key} className="p-1.5 bg-muted/40 rounded border border-border/30" data-testid={`comp-feature-${key}`}>
                  <span className="text-[9px] text-muted-foreground block truncate">{key.replace(/([A-Z])/g, " $1").trim()}</span>
                  <span className="text-xs font-mono font-medium">{typeof value === "number" ? (Math.abs(value) > 100 ? value.toFixed(1) : value.toFixed(3)) : value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function NovelRouteCard({ route, index }: { route: NovelSynthesisRoute; index: number }) {
  const confColor = CONFIDENCE_COLORS[route.synthesisConfidence] ?? CONFIDENCE_COLORS.medium;
  const noveltyPct = Math.round(route.noveltyScore * 100);

  return (
    <div className="p-3 bg-muted/50 rounded-md space-y-2.5 border border-border/50" data-testid={`synthesis-novel-${index}`}>
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-bold leading-snug">{route.method}</p>
        <div className="flex items-center gap-1.5 shrink-0">
          <Badge className={`${confColor} border-0 text-[10px]`} data-testid={`confidence-${index}`}>
            {route.synthesisConfidence}
          </Badge>
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-muted-foreground">Novelty</span>
            <div className="w-10 h-1.5 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full" style={{ width: `${noveltyPct}%` }} />
            </div>
            <span className="text-[10px] font-mono">{noveltyPct}%</span>
          </div>
        </div>
      </div>

      {route.keyInnovation && (
        <div className="p-2 bg-primary/5 border border-primary/20 rounded text-xs" data-testid={`innovation-${index}`}>
          <span className="font-semibold text-primary">Key Innovation: </span>
          {route.keyInnovation}
        </div>
      )}

      <ol className="text-xs space-y-1 ml-4 list-decimal text-muted-foreground">
        {route.steps.map((step, i) => <li key={i}>{step}</li>)}
      </ol>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px]">
        <div className="p-1.5 bg-background rounded">
          <span className="text-muted-foreground block">Temperature</span>
          <span className="font-mono">{route.temperatureProfile.length > 60 ? route.temperatureProfile.slice(0, 60) + "..." : route.temperatureProfile}</span>
        </div>
        <div className="p-1.5 bg-background rounded">
          <span className="text-muted-foreground block">Pressure</span>
          <span className="font-mono">{route.pressureProfile.length > 60 ? route.pressureProfile.slice(0, 60) + "..." : route.pressureProfile}</span>
        </div>
        <div className="p-1.5 bg-background rounded">
          <span className="text-muted-foreground block">Cooling Rate</span>
          <span className="font-mono">{route.estimatedCoolingRate}</span>
        </div>
        <div className="p-1.5 bg-background rounded">
          <span className="text-muted-foreground block">Atmosphere</span>
          <span className="font-mono">{route.atmosphere.length > 40 ? route.atmosphere.slice(0, 40) + "..." : route.atmosphere}</span>
        </div>
      </div>

      {route.expectedYieldRange && (
        <div className="text-xs">
          <span className="text-muted-foreground">Expected Yield: </span>
          <span className="font-mono">{route.expectedYieldRange}</span>
        </div>
      )}

      <div className="text-xs p-2 bg-muted/30 rounded border-l-2 border-primary/40" data-testid={`justification-${index}`}>
        <span className="font-semibold">Physics Justification: </span>
        <span className="text-muted-foreground">{route.physicsJustification}</span>
      </div>
    </div>
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
    verified: "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
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
                    <SelectItem value="verified">Verified</SelectItem>
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
        {candidate && <NovelSynthesisSection candidate={candidate} />}
        {formula && <RetrosynthesisSection formula={formula} />}
        <ReactionsSection reactions={data?.chemicalReactions ?? []} />
        <PipelineResultsSection results={data?.computationalResults ?? []} />
      </div>
    </div>
  );
}
