import { useQuery, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ArrowRight, Play, Loader2, Zap, Microscope, Target,
  CheckCircle2, GitMerge, Lightbulb, Settings2, Network,
  BookOpen, Gauge,
} from "lucide-react";

export default function CausalPhysics() {
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
  const guidanceQuery = useQuery<any[]>({ queryKey: ["/api/causal-discovery/guidance"] });

  const runMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/causal-discovery/run", { datasetSize: 60 }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/graph"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/hypotheses"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/rules"] });
      queryClient.invalidateQueries({ queryKey: ["/api/causal-discovery/guidance"] });
      queryClient.invalidateQueries({ queryKey: ["/api/theory-report"] });
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
  const interventionVars = variables.filter((v: any) => v.isIntervention);

  return (
    <div className="min-h-screen bg-background p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="gold-text synthos-heading text-3xl font-bold tracking-tight">
            Causal Physics Engine
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Discover causal mechanisms underlying superconductivity — not just correlations, but true cause-and-effect relationships.
          </p>
        </div>
        <Button
          onClick={() => runMutation.mutate()}
          disabled={runMutation.isPending}
          className="bg-[hsl(var(--gold))] hover:bg-[hsl(var(--gold-dark))] text-black font-semibold"
        >
          {runMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
          Run Causal Discovery
        </Button>
      </div>

      {statsQuery.isLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[0, 1, 2, 3].map(i => <Skeleton key={i} className="h-20" />)}
        </div>
      ) : stats ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-2xl font-bold text-[hsl(var(--gold))]">{stats.totalRuns}</div>
                <div className="text-xs text-muted-foreground">Discovery Runs</div>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-2xl font-bold text-[hsl(var(--gold))]">{stats.causalVariableCount}</div>
                <div className="text-xs text-muted-foreground">Causal Variables</div>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-2xl font-bold text-[hsl(var(--gold))]">{stats.totalHypotheses}</div>
                <div className="text-xs text-muted-foreground">Hypotheses</div>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-2xl font-bold text-[hsl(var(--gold))]">{stats.totalCausalRules}</div>
                <div className="text-xs text-muted-foreground">Causal Rules</div>
              </CardContent>
            </Card>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-lg font-bold text-[hsl(var(--gold-muted))]">{stats.ontologyNodeCount}</div>
                <div className="text-xs text-muted-foreground">Ontology Nodes</div>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-lg font-bold text-[hsl(var(--gold-muted))]">{stats.interventionVariableCount}</div>
                <div className="text-xs text-muted-foreground">Intervention Variables</div>
              </CardContent>
            </Card>
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardContent className="pt-4 pb-3 text-center">
                <div className="text-lg font-bold text-[hsl(var(--gold-muted))]">
                  {stats.latestGraphSize.nodes}N / {stats.latestGraphSize.edges}E
                </div>
                <div className="text-xs text-muted-foreground">Latest Graph (Nodes/Edges)</div>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground text-center py-4">
          No causal discovery data yet. Click "Run Causal Discovery" to begin.
        </p>
      )}

      <Tabs defaultValue="graph" className="w-full">
        <TabsList className="bg-[hsl(var(--gold)/0.05)] border border-[hsl(var(--gold)/0.2)] w-full justify-start flex-wrap h-auto gap-1 p-1">
          <TabsTrigger value="graph" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <GitMerge className="h-3.5 w-3.5 mr-1.5" />Causal Graph
          </TabsTrigger>
          <TabsTrigger value="hypotheses" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <Lightbulb className="h-3.5 w-3.5 mr-1.5" />Hypotheses
          </TabsTrigger>
          <TabsTrigger value="interventions" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <Settings2 className="h-3.5 w-3.5 mr-1.5" />Interventions
          </TabsTrigger>
          <TabsTrigger value="ontology" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <Network className="h-3.5 w-3.5 mr-1.5" />Ontology
          </TabsTrigger>
          <TabsTrigger value="rules" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <BookOpen className="h-3.5 w-3.5 mr-1.5" />Causal Rules
          </TabsTrigger>
          <TabsTrigger value="pressure" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <Gauge className="h-3.5 w-3.5 mr-1.5" />Pressure Regimes
          </TabsTrigger>
          <TabsTrigger value="guidance" className="data-[state=active]:bg-[hsl(var(--gold)/0.15)] data-[state=active]:text-[hsl(var(--gold-light))]">
            <Target className="h-3.5 w-3.5 mr-1.5" />Design Guidance
          </TabsTrigger>
        </TabsList>

        <TabsContent value="graph" className="mt-4 space-y-4">
          {graph && graph.edges && graph.edges.length > 0 ? (
            <>
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <span>Method: <span className="text-[hsl(var(--gold-light))]">{graph.method}</span></span>
                <span>Nodes: <span className="text-[hsl(var(--gold-light))]">{graph.nodes?.length ?? 0}</span></span>
                <span>Edges: <span className="text-[hsl(var(--gold-light))]">{graph.edges?.length ?? 0}</span></span>
                <span>Dataset: <span className="text-[hsl(var(--gold-light))]">{graph.datasetSize} records</span></span>
              </div>
              <Card className="border-[hsl(var(--gold)/0.25)]">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm gold-text">Discovered Causal Edges</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-2">
                      {graph.edges.map((e: any, i: number) => (
                        <div key={i} className="border border-[hsl(var(--gold)/0.15)] rounded-lg p-3 text-sm">
                          <div className="flex items-center gap-2 mb-1">
                            <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{e.source}</code>
                            <ArrowRight className="h-3 w-3 text-[hsl(var(--gold))]" />
                            <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{e.target}</code>
                            <Badge
                              variant={e.strength > 0.5 ? "default" : "secondary"}
                              className={e.strength > 0.5 ? "ml-auto bg-[hsl(var(--gold))] text-black text-xs" : "ml-auto text-xs"}
                            >
                              {(e.strength).toFixed(3)}
                            </Badge>
                            {e.validated && <CheckCircle2 className="h-4 w-4 text-green-500" />}
                          </div>
                          <p className="text-xs text-muted-foreground">{e.mechanism}</p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-12">No causal graph discovered yet. Run causal discovery first.</p>
          )}
        </TabsContent>

        <TabsContent value="hypotheses" className="mt-4 space-y-4">
          {hypotheses.length > 0 ? (
            <ScrollArea className="h-[600px]">
              <div className="space-y-3 pr-4">
                {hypotheses.map((h: any, i: number) => (
                  <Card key={h.id || i} className="border-[hsl(var(--gold)/0.25)]">
                    <CardContent className="pt-4 pb-3">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            {h.hypothesisType && (
                              <Badge
                                variant={h.hypothesisType === "design" ? "default" : "outline"}
                                className={
                                  h.hypothesisType === "design"
                                    ? "text-[10px] bg-green-600 text-white"
                                    : "text-[10px] border-[hsl(var(--gold)/0.4)] text-[hsl(var(--gold))]"
                                }
                              >
                                {h.hypothesisType === "design" ? "Design" : "Observation"}
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm font-medium">{h.statement}</p>
                          <div className="flex items-center gap-1 mt-2 flex-wrap">
                            {h.causalChain?.map((v: string, ci: number) => (
                              <span key={ci} className="flex items-center gap-0.5">
                                <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{v}</code>
                                {ci < h.causalChain.length - 1 && <ArrowRight className="h-2.5 w-2.5 text-[hsl(var(--gold))]" />}
                              </span>
                            ))}
                          </div>
                        </div>
                        <Badge
                          variant={h.confidence > 0.5 ? "default" : "secondary"}
                          className={h.confidence > 0.5 ? "bg-[hsl(var(--gold))] text-black" : ""}
                        >
                          {(h.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">{h.mechanism?.slice(0, 200)}</p>
                      {h.testableIntervention && (
                        <p className="text-xs text-[hsl(var(--gold-light))] mt-2">
                          <Microscope className="h-3 w-3 inline mr-1" />
                          {h.testableIntervention}
                        </p>
                      )}
                      {h.materialFamilies && h.materialFamilies.length > 0 && (
                        <div className="flex gap-1 mt-2 flex-wrap">
                          {h.materialFamilies.map((f: string) => (
                            <Badge key={f} variant="outline" className="text-xs border-[hsl(var(--gold)/0.3)]">{f}</Badge>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-12">No hypotheses generated yet. Run causal discovery first.</p>
          )}
        </TabsContent>

        <TabsContent value="interventions" className="mt-4 space-y-4">
          <Card className="border-[hsl(var(--gold)/0.25)]">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm gold-text flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Intervention Simulator
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Formula</label>
                  <Input
                    value={interventionFormula}
                    onChange={e => setInterventionFormula(e.target.value)}
                    className="border-[hsl(var(--gold)/0.2)] focus:border-[hsl(var(--gold)/0.5)]"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Variable</label>
                  <select
                    className="w-full border border-[hsl(var(--gold)/0.2)] rounded-md px-3 py-2 text-sm bg-background focus:border-[hsl(var(--gold)/0.5)] focus:outline-none"
                    value={interventionVar}
                    onChange={e => setInterventionVar(e.target.value)}
                  >
                    {interventionVars.map((v: any) => (
                      <option key={v.name} value={v.name}>{v.name} ({v.unit})</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">New Value</label>
                  <Input
                    value={interventionValue}
                    onChange={e => setInterventionValue(e.target.value)}
                    type="number"
                    className="border-[hsl(var(--gold)/0.2)] focus:border-[hsl(var(--gold)/0.5)]"
                  />
                </div>
              </div>
              <Button
                onClick={() => interventionMutation.mutate()}
                disabled={interventionMutation.isPending}
                className="bg-[hsl(var(--gold))] hover:bg-[hsl(var(--gold-dark))] text-black font-semibold"
              >
                {interventionMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Zap className="h-4 w-4 mr-2" />}
                Simulate Intervention
              </Button>

              {interventionMutation.data && (() => {
                const r: any = interventionMutation.data;
                return (
                  <div className="mt-4 border border-[hsl(var(--gold)/0.25)] rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-muted-foreground">Tc: {r.tcOriginal}K</span>
                      <ArrowRight className="h-3 w-3 text-[hsl(var(--gold))]" />
                      <span className={r.tcChange > 0 ? "text-green-500 font-semibold" : r.tcChange < 0 ? "text-red-500 font-semibold" : "font-semibold"}>
                        {r.tcNew}K ({r.tcChange > 0 ? "+" : ""}{r.tcChange}K)
                      </span>
                    </div>
                    {r.causalPathway && r.causalPathway.length > 1 && (
                      <div className="text-xs text-muted-foreground">
                        Causal pathway: <span className="text-[hsl(var(--gold-light))]">{r.causalPathway.join(" → ")}</span>
                      </div>
                    )}
                    {r.effects && r.effects.length > 0 && (
                      <div className="space-y-1 mt-2">
                        <p className="text-xs font-medium text-[hsl(var(--gold))]">Propagated Effects:</p>
                        <div className="border border-[hsl(var(--gold)/0.1)] rounded overflow-hidden">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="bg-[hsl(var(--gold)/0.05)]">
                                <th className="text-left px-3 py-1.5 font-medium">Variable</th>
                                <th className="text-left px-3 py-1.5 font-medium">Original</th>
                                <th className="text-left px-3 py-1.5 font-medium">New</th>
                                <th className="text-left px-3 py-1.5 font-medium">Change</th>
                              </tr>
                            </thead>
                            <tbody>
                              {r.effects.slice(0, 8).map((eff: any, i: number) => (
                                <tr key={i} className="border-t border-[hsl(var(--gold)/0.08)]">
                                  <td className="px-3 py-1.5">
                                    <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{eff.variable}</code>
                                  </td>
                                  <td className="px-3 py-1.5">{eff.originalValue?.toFixed(2)}</td>
                                  <td className="px-3 py-1.5">{eff.newValue?.toFixed(2)}</td>
                                  <td className={`px-3 py-1.5 font-medium ${eff.changePercent > 0 ? "text-green-500" : "text-red-500"}`}>
                                    {eff.changePercent > 0 ? "+" : ""}{eff.changePercent}%
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </CardContent>
          </Card>

          <Card className="border-[hsl(var(--gold)/0.25)]">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm gold-text flex items-center gap-2">
                <Microscope className="h-4 w-4" />
                Counterfactual Simulator
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Variable</label>
                  <select
                    className="w-full border border-[hsl(var(--gold)/0.2)] rounded-md px-3 py-2 text-sm bg-background focus:border-[hsl(var(--gold)/0.5)] focus:outline-none"
                    value={counterfactualVar}
                    onChange={e => setCounterfactualVar(e.target.value)}
                  >
                    {variables.map((v: any) => (
                      <option key={v.name} value={v.name}>{v.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">Modification %</label>
                  <Input
                    value={counterfactualPct}
                    onChange={e => setCounterfactualPct(e.target.value)}
                    type="number"
                    className="border-[hsl(var(--gold)/0.2)] focus:border-[hsl(var(--gold)/0.5)]"
                  />
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={() => counterfactualMutation.mutate()}
                    disabled={counterfactualMutation.isPending}
                    className="bg-[hsl(var(--gold))] hover:bg-[hsl(var(--gold-dark))] text-black font-semibold"
                  >
                    {counterfactualMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Microscope className="h-4 w-4 mr-2" />}
                    Ask "What If?"
                  </Button>
                </div>
              </div>

              {counterfactualMutation.data && (() => {
                const r: any = counterfactualMutation.data;
                return (
                  <div className="mt-3 border border-[hsl(var(--gold)/0.25)] rounded-lg p-4 space-y-3">
                    <p className="text-sm font-medium">{r.question}</p>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-muted-foreground">Tc: {r.originalTc?.toFixed(1)}K</span>
                      <ArrowRight className="h-3 w-3 text-[hsl(var(--gold))]" />
                      <span className={r.tcDelta > 0 ? "text-green-500 font-semibold" : r.tcDelta < 0 ? "text-red-500 font-semibold" : "font-semibold"}>
                        {r.counterfactualTc?.toFixed(1)}K ({r.tcDelta > 0 ? "+" : ""}{r.tcDelta?.toFixed(1)}K, {r.tcDeltaPercent > 0 ? "+" : ""}{r.tcDeltaPercent}%)
                      </span>
                    </div>
                    <p className="text-xs text-[hsl(var(--gold-light))]">{r.designImplication}</p>
                    {r.propagatedEffects && r.propagatedEffects.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs font-medium text-[hsl(var(--gold))]">Downstream effects:</p>
                        <div className="flex flex-wrap gap-2 mt-1">
                          {r.propagatedEffects.slice(0, 6).map((pe: any, i: number) => (
                            <Badge key={i} variant="outline" className="text-xs border-[hsl(var(--gold)/0.3)]">
                              {pe.variable}: {pe.change > 0 ? "+" : ""}{pe.change?.toFixed(1)}%
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ontology" className="mt-4 space-y-4">
          {ontology.length > 0 ? (
            <ScrollArea className="h-[600px]">
              <div className="space-y-4 pr-4">
                {["atomic_structure", "electronic_structure", "phonon_properties", "pairing_interactions", "thermodynamic_conditions", "superconducting_properties", "topological"].map(cat => {
                  const nodes = ontology.filter((n: any) => n.category === cat);
                  if (nodes.length === 0) return null;
                  return (
                    <Card key={cat} className="border-[hsl(var(--gold)/0.25)]">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm gold-text capitalize">{cat.replace(/_/g, " ")}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {nodes.map((n: any) => (
                            <div key={n.variable} className="border border-[hsl(var(--gold)/0.15)] rounded-lg p-3 text-xs">
                              <div className="flex items-center justify-between mb-1">
                                <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono font-medium">{n.variable}</code>
                                <Badge variant="secondary" className="text-xs">Level {n.level}</Badge>
                              </div>
                              {n.parents.length > 0 && (
                                <div className="text-muted-foreground mt-1">
                                  Parents: {n.parents.map((p: string) => (
                                    <code key={p} className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono mr-1">{p}</code>
                                  ))}
                                </div>
                              )}
                              {n.children.length > 0 && (
                                <div className="text-muted-foreground mt-1">
                                  Children: {n.children.map((c: string) => (
                                    <code key={c} className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono mr-1">{c}</code>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-12">Loading ontology...</p>
          )}
        </TabsContent>

        <TabsContent value="rules" className="mt-4 space-y-3">
          {rules.length > 0 ? (
            <ScrollArea className="h-[600px]">
              <div className="space-y-3 pr-4">
                {rules.map((r: any, i: number) => (
                  <Card key={i} className="border-[hsl(var(--gold)/0.25)]">
                    <CardContent className="pt-4 pb-3">
                      <div className="flex items-center gap-2 text-sm mb-1">
                        <Badge variant="outline" className="font-mono text-xs border-[hsl(var(--gold)/0.3)]">{r.antecedent}</Badge>
                        <ArrowRight className="h-3 w-3 text-[hsl(var(--gold))]" />
                        <Badge variant="outline" className="font-mono text-xs border-[hsl(var(--gold)/0.3)]">{r.consequent}</Badge>
                        <span className="ml-auto text-xs text-muted-foreground">strength: <span className="text-[hsl(var(--gold-light))]">{r.strength?.toFixed(3)}</span></span>
                      </div>
                      <p className="text-xs text-muted-foreground">{r.mechanism}</p>
                      {r.validatedAcross && r.validatedAcross.length > 0 && (
                        <div className="flex gap-1 mt-2 flex-wrap">
                          <span className="text-xs text-muted-foreground">Validated:</span>
                          {r.validatedAcross.map((f: string) => (
                            <Badge key={f} variant="secondary" className="text-xs">{f}</Badge>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-12">No causal rules extracted yet. Run causal discovery first.</p>
          )}
        </TabsContent>

        <TabsContent value="pressure" className="mt-4 space-y-4">
          {runMutation.data ? (() => {
            const pc: any = (runMutation.data as any).pressureComparison;
            if (!pc) return <p className="text-sm text-muted-foreground text-center py-12">No pressure comparison data. Run discovery first.</p>;
            return (
              <>
                <Card className="border-[hsl(var(--gold)/0.25)]">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm gold-text">Decompression Insight</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm">{pc.decompressionInsight}</p>
                  </CardContent>
                </Card>

                {pc.survivingMechanisms && pc.survivingMechanisms.length > 0 && (
                  <Card className="border-[hsl(var(--gold)/0.25)]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm gold-text">Mechanisms Surviving Across Pressure Regimes</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {pc.survivingMechanisms.map((m: any, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-sm border border-[hsl(var(--gold)/0.15)] rounded-lg p-3">
                            <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{m.source}</code>
                            <ArrowRight className="h-3 w-3 text-[hsl(var(--gold))]" />
                            <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{m.target}</code>
                            <span className="ml-auto text-xs text-muted-foreground">
                              ambient: <span className="text-[hsl(var(--gold-light))]">{m.ambientStrength?.toFixed(3)}</span> | high-P: <span className="text-[hsl(var(--gold-light))]">{m.hpStrength?.toFixed(3)}</span>
                            </span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {pc.newMechanisms && pc.newMechanisms.length > 0 && (
                  <Card className="border-[hsl(var(--gold)/0.25)]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm gold-text">Regime-Specific Mechanisms</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {pc.newMechanisms.map((m: any, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-sm border border-[hsl(var(--gold)/0.15)] rounded-lg p-3">
                            <Badge
                              variant={m.regime === "ambient" ? "default" : "secondary"}
                              className={m.regime === "ambient" ? "bg-[hsl(var(--gold))] text-black text-xs" : "text-xs"}
                            >
                              {m.regime}
                            </Badge>
                            <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{m.source}</code>
                            <ArrowRight className="h-3 w-3 text-[hsl(var(--gold))]" />
                            <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{m.target}</code>
                            <span className="ml-auto text-xs text-[hsl(var(--gold-light))]">{m.strength?.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            );
          })() : (
            <p className="text-sm text-muted-foreground text-center py-12">Run causal discovery to see pressure regime comparison data.</p>
          )}
        </TabsContent>

        <TabsContent value="guidance" className="mt-4 space-y-4">
          {guidanceQuery.isLoading ? (
            <div className="space-y-3">
              {[0, 1, 2].map(i => <Skeleton key={i} className="h-24" />)}
            </div>
          ) : (guidanceQuery.data ?? []).length > 0 ? (
            <Card className="border-[hsl(var(--gold)/0.25)]">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm gold-text flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Material Design Recommendations
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  Ranked by causal impact on Tc. Direction determined via partial correlation (controlling for other intervention variables).
                </p>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[500px]">
                  <div className="space-y-3 pr-4">
                    {(guidanceQuery.data ?? []).map((g: any, i: number) => (
                      <div key={i} className="border border-[hsl(var(--gold)/0.2)] rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Badge
                              variant={g.direction === "maximize" ? "default" : "secondary"}
                              className={g.direction === "maximize" ? "bg-[hsl(var(--gold))] text-black text-xs" : "text-xs"}
                            >
                              #{g.rank} {g.direction === "maximize" ? "↑ Maximize" : "↓ Minimize"}
                            </Badge>
                            <span className="text-sm font-medium">{g.variableLabel || g.variable}</span>
                          </div>
                          <Badge variant="outline" className="font-mono text-xs border-[hsl(var(--gold)/0.3)]">
                            Impact: {g.causalImpactOnTc}
                          </Badge>
                        </div>
                        <p className="text-sm text-green-500 font-medium mb-1">{g.recommendation}</p>
                        <div className="text-xs text-muted-foreground flex items-center gap-1">
                          <span>Pathway:</span>
                          <code className="bg-[hsl(var(--gold)/0.08)] text-[hsl(var(--gold-light))] px-1.5 py-0.5 rounded text-xs font-mono">{g.mechanism}</code>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-12">No design guidance available. Run causal discovery first to generate recommendations.</p>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
