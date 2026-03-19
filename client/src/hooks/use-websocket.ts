import { useState, useEffect, useCallback, useRef, useSyncExternalStore } from "react";
import { queryClient } from "../lib/queryClient";

export interface WSMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface ThoughtMessage {
  text: string;
  category: "strategy" | "discovery" | "stagnation" | "milestone" | "computation";
  timestamp: string;
}

export type EngineTempo = "excited" | "exploring" | "contemplating";

interface GlobalWSState {
  connected: boolean;
  messages: WSMessage[];
  messageCount: number;
  engineState: string;
  activeTasks: string[];
  thoughts: ThoughtMessage[];
  tempo: EngineTempo;
  statusMessage: string;
}

const globalState: GlobalWSState = {
  connected: false,
  messages: [],
  messageCount: 0,
  engineState: "stopped",
  activeTasks: [],
  thoughts: [],
  tempo: "exploring",
  statusMessage: "",
};

let listeners: Set<() => void> = new Set();
let wsInstance: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
let wsInitialized = false;
let statusFetched = false;

function notifyListeners() {
  listeners.forEach((fn) => fn());
}

function updateState(partial: Partial<GlobalWSState>) {
  Object.assign(globalState, partial);
  notifyListeners();
}

function connectWS() {
  if (wsInstance?.readyState === WebSocket.OPEN) return;

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
  wsInstance = ws;

  ws.onopen = () => {
    reconnectAttempts = 0;
    updateState({ connected: true });
    // Flush stale query cache on every reconnect so UI immediately reflects current server state.
    // Without this, queries cached from before a server restart stay stale indefinitely.
    queryClient.invalidateQueries();
    // Seed the activity feed with recent log history so it's not empty on connect/reconnect
    fetch("/api/research-logs?limit=20")
      .then((r) => r.json())
      .then((logs: any[]) => {
        if (!Array.isArray(logs) || logs.length === 0) return;
        // Convert DB log format to WSMessage format, oldest first
        const historical: WSMessage[] = logs.reverse().map((log) => ({
          type: "log",
          data: { phase: log.phase, event: log.event, detail: log.detail, dataSource: log.dataSource },
          timestamp: log.timestamp ?? new Date().toISOString(),
        }));
        // Merge: historical first, then any live messages already received, deduplicate by timestamp+event
        const seen = new Set<string>();
        const merged = [...historical, ...globalState.messages].filter((m) => {
          const key = `${m.timestamp}::${m.data?.event ?? m.type}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        globalState.messages = merged.slice(-100);
        notifyListeners();
      })
      .catch(() => {});
  };

  ws.onmessage = (event) => {
    try {
      const msg: WSMessage = JSON.parse(event.data);

      globalState.messages = [...globalState.messages.slice(-100), msg];
      globalState.messageCount++;

      if (msg.type === "status" || msg.type === "engineState") {
        globalState.engineState = msg.data.state ?? msg.data;
        if (msg.data.activeTasks) {
          globalState.activeTasks = msg.data.activeTasks;
        }
      }

      if (msg.type === "taskStart") {
        if (!globalState.activeTasks.includes(msg.data.task)) {
          globalState.activeTasks = [...globalState.activeTasks, msg.data.task];
        }
      }

      if (msg.type === "taskEnd") {
        globalState.activeTasks = globalState.activeTasks.filter((t) => t !== msg.data.task);
      }

      if (msg.type === "thought") {
        globalState.thoughts = [
          ...globalState.thoughts.slice(-30),
          {
            text: msg.data.text,
            category: msg.data.category ?? "strategy",
            timestamp: msg.timestamp,
          },
        ];
      }

      if (msg.type === "tempoChange") {
        globalState.tempo = msg.data.tempo ?? "exploring";
      }

      if (msg.type === "statusMessage") {
        globalState.statusMessage = msg.data.message ?? "";
        if (msg.data.tempo) globalState.tempo = msg.data.tempo;
      }

      notifyListeners();
    } catch (e) {
      console.warn("WebSocket parse error:", e);
    }
  };

  ws.onclose = () => {
    globalState.connected = false;
    notifyListeners();
    wsInstance = null;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectAttempts = Math.min(reconnectAttempts + 1, 5);
    const backoffMs = Math.min(3000 * Math.pow(2, reconnectAttempts - 1), 30000);
    reconnectTimer = setTimeout(connectWS, backoffMs);
  };

  ws.onerror = () => {
    ws.close();
  };
}

function initWS() {
  if (wsInitialized) return;
  wsInitialized = true;

  if (!statusFetched) {
    statusFetched = true;
    fetch("/api/engine/status")
      .then((r) => r.json())
      .then((status) => {
        if (status.state) globalState.engineState = status.state;
        if (status.activeTasks) globalState.activeTasks = status.activeTasks;
        if (status.tempo) globalState.tempo = status.tempo;
        if (status.statusMessage) globalState.statusMessage = status.statusMessage;
        notifyListeners();
      })
      .catch(() => {});
  }

  connectWS();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  initWS();
  return () => {
    listeners.delete(listener);
  };
}

let snapshotRef = { ...globalState };

function getSnapshot() {
  if (
    snapshotRef.connected !== globalState.connected ||
    snapshotRef.messages !== globalState.messages ||
    snapshotRef.messageCount !== globalState.messageCount ||
    snapshotRef.engineState !== globalState.engineState ||
    snapshotRef.activeTasks !== globalState.activeTasks ||
    snapshotRef.thoughts !== globalState.thoughts ||
    snapshotRef.tempo !== globalState.tempo ||
    snapshotRef.statusMessage !== globalState.statusMessage
  ) {
    snapshotRef = { ...globalState };
  }
  return snapshotRef;
}

export function useWebSocket() {
  const state = useSyncExternalStore(subscribe, getSnapshot);

  const clearMessages = useCallback(() => {
    globalState.messages = [];
    notifyListeners();
  }, []);

  return {
    connected: state.connected,
    messages: state.messages,
    messageCount: state.messageCount,
    engineState: state.engineState,
    activeTasks: state.activeTasks,
    thoughts: state.thoughts,
    tempo: state.tempo,
    statusMessage: state.statusMessage,
    clearMessages,
  };
}
