import { useState, useEffect, useCallback, useRef, useSyncExternalStore } from "react";

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
  engineState: string;
  activeTasks: string[];
  thoughts: ThoughtMessage[];
  tempo: EngineTempo;
  statusMessage: string;
}

const globalState: GlobalWSState = {
  connected: false,
  messages: [],
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
  };

  ws.onmessage = (event) => {
    try {
      const msg: WSMessage = JSON.parse(event.data);

      globalState.messages = [...globalState.messages.slice(-100), msg];

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
    engineState: state.engineState,
    activeTasks: state.activeTasks,
    thoughts: state.thoughts,
    tempo: state.tempo,
    statusMessage: state.statusMessage,
    clearMessages,
  };
}
