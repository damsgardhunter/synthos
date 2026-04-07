import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { queryClient } from "@/lib/queryClient";
import { useWebSocket } from "@/hooks/use-websocket";
import type { NovelPrediction } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { PaginationBar } from "@/components/ui/pagination-bar";
import { Link } from "wouter";
import { FlaskConical, Zap, Star, CheckCircle2, Clock, Eye, Atom, ExternalLink, BookOpen } from "lucide-react";

const PAGE_SIZE = 12;

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: any }> = {
  "predicted": {
    label: "AI Prediction",
    color: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
    icon: Clock,
  },
  "under_review": {
    label: "Under Review",
    color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
    icon: Eye,
  },
  "literature-reported": {
    label: "Literature-Reported",
    color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
    icon: BookOpen,
  },
  "synthesized": {
    label: "Literature-Reported",
    color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
    icon: BookOpen,
  },
};

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = confidence * 100;
  const color = pct >= 80 ? "bg-green-500" : pct >= 60 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Model Confidence</span>
        <span className="text-xs font-mono font-bold">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function PredictionCard({ prediction }: { prediction: NovelPrediction }) {
  const statusCfg = STATUS_CONFIG[prediction.status] ?? STATUS_CONFIG["predicted"];
  const StatusIcon = statusCfg.icon;
  const props = prediction.predictedProperties as Record<string, any> ?? {};

  return (
    <Card data-testid={`prediction-card-${prediction.id}`} className="flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <div>
            <CardTitle className="text-base leading-snug">{prediction.name}</CardTitle>
            <p className="text-sm font-mono text-primary mt-1">{prediction.formula}</p>
          </div>
          <Badge className={`${statusCfg.color} border-0 flex items-center gap-1`}>
            <StatusIcon className="h-3 w-3" />
            {statusCfg.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="flex-1 space-y-4">
        {(prediction.status === "literature-reported" || prediction.status === "synthesized") && (
          <div className="flex items-center gap-1.5 text-[10px] text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/30 px-2 py-1 rounded">
            <BookOpen className="h-3 w-3" />
            Source: Published literature (not a platform prediction)
          </div>
        )}
        <p className="text-sm text-muted-foreground leading-relaxed">{prediction.notes}</p>

        <div className="p-3 bg-muted/50 rounded-md">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Target Application</p>
          <p className="text-sm font-medium">{prediction.targetApplication}</p>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Predicted Properties</p>
          <div className="space-y-1.5">
            {Object.entries(props).slice(0, 5).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between text-sm border-b border-border pb-1.5 last:border-0">
                <span className="text-muted-foreground capitalize">{k.replace(/([A-Z])/g, " $1").replace(/_/g, " ")}</span>
                <span className="font-mono font-medium text-foreground">
                  {typeof v === "boolean" ? (v ? "Yes" : "No") : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <ConfidenceBar confidence={prediction.confidence} />

        <Link href={`/candidate/${encodeURIComponent(prediction.formula)}`} data-testid={`link-discovery-profile-${prediction.id}`}>
          <span className="inline-flex items-center gap-1 text-xs text-primary hover:underline cursor-pointer">
            <ExternalLink className="h-3 w-3" />
            View Candidate Profile
          </span>
        </Link>
      </CardContent>
    </Card>
  );
}

function PagedSection({ items, isLoading, skeletonCount = 2 }: { items: NovelPrediction[]; isLoading: boolean; skeletonCount?: number }) {
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const paged = items.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: skeletonCount }).map((_, i) => <Skeleton key={i} className="h-64" />)}
      </div>
    );
  }
  if (items.length === 0) return null;
  return (
    <>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {paged.map(p => <PredictionCard key={p.id} prediction={p} />)}
      </div>
      <PaginationBar page={safePage} totalPages={totalPages} onPageChange={setPage} />
    </>
  );
}

export default function NovelDiscovery() {
  const { data: predictions, isLoading } = useQuery<NovelPrediction[]>({
    queryKey: ["/api/novel-predictions"],
  });

  const ws = useWebSocket();

  useEffect(() => {
    const last = ws.messages[ws.messages.length - 1];
    if (!last) return;
    // "log" and "progress" fire many times/sec — excluded to prevent request storms
    const relevantTypes = new Set(["phaseUpdate", "prediction", "cycleEnd"]);
    if (relevantTypes.has(last.type)) {
      queryClient.invalidateQueries({ queryKey: ["/api/novel-predictions"] });
    }
  }, [ws.messageCount]);

  const literatureReported = predictions?.filter(p => p.status === "literature-reported" || p.status === "synthesized") ?? [];
  const underReview = predictions?.filter(p => p.status === "under_review") ?? [];
  const predicted = predictions?.filter(p => p.status === "predicted") ?? [];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <FlaskConical className="h-6 w-6 text-primary" />
          Novel Material Discovery
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          AI-generated predictions for new materials with transformative properties — from superconductors to ultra-hard compounds.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Card data-testid="stat-literature">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <BookOpen className="h-4 w-4 text-emerald-500" />
              <span className="text-xs text-muted-foreground">Literature-Reported</span>
            </div>
            <div className="text-2xl font-bold font-mono">{literatureReported.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="stat-under-review">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <Eye className="h-4 w-4 text-yellow-500" />
              <span className="text-xs text-muted-foreground">Under Review</span>
            </div>
            <div className="text-2xl font-bold font-mono">{underReview.length}</div>
          </CardContent>
        </Card>
        <Card data-testid="stat-predictions">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="h-4 w-4 text-blue-500" />
              <span className="text-xs text-muted-foreground">Awaiting Synthesis</span>
            </div>
            <div className="text-2xl font-bold font-mono">{predicted.length}</div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-emerald-500" />
          <h2 className="text-base font-semibold">Literature-Reported Materials</h2>
          <Badge variant="secondary" className="bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300 border-0">{literatureReported.length}</Badge>
        </div>
        <p className="text-xs text-muted-foreground">Known materials from published research, included as reference data. Not predictions by this platform.</p>
        <PagedSection items={literatureReported} isLoading={isLoading} />
        {!isLoading && literatureReported.length === 0 && (
          <Card><CardContent className="py-6 text-center text-muted-foreground text-sm">No literature-reported materials indexed yet</CardContent></Card>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Eye className="h-4 w-4 text-yellow-500" />
          <h2 className="text-base font-semibold">Under Scientific Review</h2>
          <Badge variant="secondary" className="bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300 border-0">{underReview.length}</Badge>
        </div>
        <PagedSection items={underReview} isLoading={isLoading} />
        {!isLoading && underReview.length === 0 && (
          <Card><CardContent className="py-6 text-center text-muted-foreground text-sm">No materials under review</CardContent></Card>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Atom className="h-4 w-4 text-blue-500" />
          <h2 className="text-base font-semibold">Predicted — Awaiting Synthesis</h2>
          <Badge variant="secondary" className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 border-0">{predicted.length}</Badge>
        </div>
        <PagedSection items={predicted} isLoading={isLoading} />
        {!isLoading && predicted.length === 0 && (
          <Card><CardContent className="py-6 text-center text-muted-foreground text-sm">No pending predictions</CardContent></Card>
        )}
      </div>

      <Card className="border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Zap className="h-4 w-4 text-primary" />
            Discovery Methodology
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { step: "1", title: "High-Throughput Screening", desc: "DFT calculations evaluate millions of candidate compositions for thermodynamic stability" },
              { step: "2", title: "ML Property Prediction", desc: "Graph neural networks predict band gaps, mechanical properties, and superconducting Tc" },
              { step: "3", title: "Synthesizability Filter", desc: "Machine learning models predict likelihood of successful laboratory synthesis" },
              { step: "4", title: "Experimental Validation", desc: "Top candidates sent to partner labs for synthesis and characterization" },
            ].map(item => (
              <div key={item.step} className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="h-6 w-6 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center">{item.step}</div>
                  <span className="text-sm font-medium">{item.title}</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed pl-8">{item.desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
