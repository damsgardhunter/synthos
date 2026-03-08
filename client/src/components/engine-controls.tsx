import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Play, Square, Pause, Activity,
  Loader2, Zap, Wifi, WifiOff, Star
} from "lucide-react";
import type { WSMessage } from "@/hooks/use-websocket";

interface EngineControlsProps {
  engineState: string;
  activeTasks: string[];
  connected: boolean;
  messages: WSMessage[];
}

export function EngineControls({ engineState, activeTasks, connected, messages }: EngineControlsProps) {
  const startMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/engine/start"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/engine/status"] });
    },
  });

  const stopMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/engine/stop"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/engine/status"] });
    },
  });

  const pauseMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/engine/pause"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/engine/status"] });
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/engine/resume"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/engine/status"] });
    },
  });

  const isLoading = startMutation.isPending || stopMutation.isPending || pauseMutation.isPending || resumeMutation.isPending;

  const stateColor: Record<string, string> = {
    running: "text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950",
    paused: "text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-950",
    stopped: "text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950",
  };

  const recentEvents = messages
    .filter((m) => m.type === "log" || m.type === "prediction" || m.type === "insight" || m.type === "milestone")
    .slice(-10)
    .reverse();

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              Learning Engine
            </div>
            <div className="flex items-center gap-2">
              {connected ? (
                <Badge variant="secondary" className="text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950 border-0 text-xs" data-testid="badge-ws-connected">
                  <Wifi className="h-3 w-3 mr-1" />Live
                </Badge>
              ) : (
                <Badge variant="secondary" className="text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950 border-0 text-xs" data-testid="badge-ws-disconnected">
                  <WifiOff className="h-3 w-3 mr-1" />Offline
                </Badge>
              )}
              <Badge variant="secondary" className={`border-0 text-xs ${stateColor[engineState] || stateColor.stopped}`} data-testid="badge-engine-state">
                {engineState === "running" && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
                {engineState.charAt(0).toUpperCase() + engineState.slice(1)}
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 flex-wrap">
            {engineState === "stopped" && (
              <Button
                size="sm"
                onClick={() => startMutation.mutate()}
                disabled={isLoading}
                data-testid="button-start-engine"
              >
                {startMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
                Start Learning
              </Button>
            )}
            {engineState === "running" && (
              <>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => pauseMutation.mutate()}
                  disabled={isLoading}
                  data-testid="button-pause-engine"
                >
                  {pauseMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Pause className="h-4 w-4 mr-1" />}
                  Pause
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => stopMutation.mutate()}
                  disabled={isLoading}
                  data-testid="button-stop-engine"
                >
                  {stopMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Square className="h-4 w-4 mr-1" />}
                  Stop
                </Button>
              </>
            )}
            {engineState === "paused" && (
              <>
                <Button
                  size="sm"
                  onClick={() => resumeMutation.mutate()}
                  disabled={isLoading}
                  data-testid="button-resume-engine"
                >
                  {resumeMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
                  Resume
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => stopMutation.mutate()}
                  disabled={isLoading}
                  data-testid="button-stop-engine-paused"
                >
                  {stopMutation.isPending ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Square className="h-4 w-4 mr-1" />}
                  Stop
                </Button>
              </>
            )}
          </div>

          {activeTasks.length > 0 && (
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <span className="text-xs text-muted-foreground">Active:</span>
              {activeTasks.map((task) => (
                <Badge key={task} variant="secondary" className="text-xs border-0 bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300" data-testid={`badge-task-${task.toLowerCase().replace(/ /g, "-")}`}>
                  <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                  {task}
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {recentEvents.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="h-4 w-4 text-primary" />
              Live Activity Feed
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-48">
              <div className="divide-y divide-border">
                {recentEvents.map((msg, i) => {
                  const isMilestone = msg.type === "milestone";
                  return (
                    <div
                      key={i}
                      className={`px-4 py-2.5 flex items-start gap-2 ${isMilestone ? "bg-amber-50/80 dark:bg-amber-950/30 border-l-2 border-amber-500" : ""}`}
                      data-testid={`live-event-${i}`}
                    >
                      {isMilestone ? (
                        <Star className="h-3.5 w-3.5 text-amber-500 mt-0.5 flex-shrink-0 fill-amber-500" />
                      ) : (
                        <Zap className="h-3 w-3 text-primary mt-0.5 flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <span className={`text-xs font-medium ${isMilestone ? "text-amber-700 dark:text-amber-300" : ""}`}>
                          {isMilestone
                            ? msg.data.title
                            : msg.type === "log"
                              ? msg.data.event
                              : msg.type === "prediction"
                                ? `New prediction: ${msg.data.name}`
                                : `Insight discovered`}
                        </span>
                        {isMilestone && msg.data.significance > 0 && (
                          <span className="ml-1.5 text-amber-500 text-xs">
                            {Array.from({ length: msg.data.significance }, (_, j) => (
                              <Star key={j} className="h-2.5 w-2.5 inline fill-amber-400 text-amber-400" />
                            ))}
                          </span>
                        )}
                        <p className="text-xs text-muted-foreground mt-0.5 break-words line-clamp-3">
                          {isMilestone
                            ? msg.data.description
                            : msg.type === "log"
                              ? msg.data.detail
                              : msg.type === "prediction"
                                ? `${msg.data.formula} (${(msg.data.confidence * 100).toFixed(0)}% confidence)`
                                : msg.data.insights?.[0]}
                        </p>
                      </div>
                      <span className="text-xs text-muted-foreground font-mono flex-shrink-0">
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      </span>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
