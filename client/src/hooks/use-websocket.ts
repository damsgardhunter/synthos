import { useState, useEffect, useCallback, useRef } from "react";

export interface WSMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface ThoughtMessage {
  text: string;
  category: "strategy" | "discovery" | "stagnation" | "milestone";
  timestamp: string;
}

export type EngineTempo = "excited" | "exploring" | "contemplating";

export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [engineState, setEngineState] = useState<string>("stopped");
  const [activeTasks, setActiveTasks] = useState<string[]>([]);
  const [thoughts, setThoughts] = useState<ThoughtMessage[]>([]);
  const [tempo, setTempo] = useState<EngineTempo>("exploring");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    fetch("/api/engine/status")
      .then((r) => r.json())
      .then((status) => {
        if (status.state) setEngineState(status.state);
        if (status.activeTasks) setActiveTasks(status.activeTasks);
        if (status.tempo) setTempo(status.tempo);
        if (status.statusMessage) setStatusMessage(status.statusMessage);
      })
      .catch(() => {});
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        setMessages((prev) => [...prev.slice(-100), msg]);

        if (msg.type === "status" || msg.type === "engineState") {
          setEngineState(msg.data.state ?? msg.data);
          if (msg.data.activeTasks) {
            setActiveTasks(msg.data.activeTasks);
          }
        }

        if (msg.type === "taskStart") {
          setActiveTasks((prev) =>
            prev.includes(msg.data.task) ? prev : [...prev, msg.data.task]
          );
        }

        if (msg.type === "taskEnd") {
          setActiveTasks((prev) => prev.filter((t) => t !== msg.data.task));
        }

        if (msg.type === "thought") {
          setThoughts((prev) => [
            ...prev.slice(-30),
            {
              text: msg.data.text,
              category: msg.data.category ?? "strategy",
              timestamp: msg.timestamp,
            },
          ]);
        }

        if (msg.type === "tempoChange") {
          setTempo(msg.data.tempo ?? "exploring");
        }

        if (msg.type === "statusMessage") {
          setStatusMessage(msg.data.message ?? "");
          if (msg.data.tempo) setTempo(msg.data.tempo);
        }
      } catch (e) {
        console.warn("WebSocket parse error:", e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      wsRef.current?.close();
    };
  }, [connect]);

  const clearMessages = useCallback(() => setMessages([]), []);

  return {
    connected,
    messages,
    engineState,
    activeTasks,
    thoughts,
    tempo,
    statusMessage,
    clearMessages,
  };
}
