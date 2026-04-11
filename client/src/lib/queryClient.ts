import { QueryClient, QueryFunction } from "@tanstack/react-query";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  const _start = Date.now();
  const res = await fetch(url, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });
  // Track all mutation timings — catches slow /api/client-errors POSTs (e.g. 16s)
  // without going through reportClientError (which would create a feedback loop).
  if (url !== "/api/network-metrics") {
    pushNetMetric({ endpoint: url, status: res.status, durationMs: Date.now() - _start });
  }

  await throwIfResNotOk(res);
  return res;
}

// ── Network metrics ring buffer ───────────────────────────────────────────────
// Every fetch timing is pushed here regardless of success/failure.
// Flushed to /api/network-metrics every 30s so the monitor can see the full
// network table — including slow 204s, repeated identical requests, flooding.
interface NetMetricEntry {
  endpoint: string;
  status: number;
  durationMs: number;
  sizeBytes?: number;
}
const _netMetricsBuf: NetMetricEntry[] = [];
function pushNetMetric(e: NetMetricEntry) {
  _netMetricsBuf.push(e);
  if (_netMetricsBuf.length > 200) _netMetricsBuf.shift();
}
function flushNetMetrics() {
  if (_netMetricsBuf.length === 0) return;
  const batch = _netMetricsBuf.splice(0, _netMetricsBuf.length);
  fetch("/api/network-metrics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(batch),
  }).catch(() => {});
}
setInterval(flushNetMetrics, 30_000);

// Per-endpoint throttle: don't report the same endpoint+status more than once per 60s.
// This prevents a storm of identical reports when many cycleEnd queries all fail at once.
const _reportedRecently = new Map<string, number>();
const REPORT_THROTTLE_MS = 60_000;

/** Fire-and-forget error report to /api/client-errors. Never throws. */
function reportClientError(payload: {
  type: string;
  page: string;
  message: string;
  endpoint?: string;
  statusCode?: number;
  durationMs?: number;
}) {
  // Never report failures of the error-reporter itself (feedback loop).
  if (payload.endpoint === "/api/client-errors") return;

  // Throttle: same endpoint+statusCode reported at most once per 60s.
  const throttleKey = `${payload.endpoint}:${payload.statusCode ?? "net"}`;
  const lastSent = _reportedRecently.get(throttleKey);
  const now = Date.now();
  if (lastSent !== undefined && now - lastSent < REPORT_THROTTLE_MS) return;
  _reportedRecently.set(throttleKey, now);

  try {
    fetch("/api/client-errors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {});
  } catch {
    // ignore — reporting must never affect the app
  }
}

const SLOW_THRESHOLD_MS = 5000;

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const endpoint = queryKey.join("/") as string;
    const page = window.location.pathname;
    const start = Date.now();

    let res: Response;
    try {
      res = await fetch(endpoint, {
        credentials: "include",
        cache: "no-cache", // always revalidate — prevents browser serving 30-min-old cached responses
      });
    } catch (networkErr) {
      const durationMs = Date.now() - start;
      pushNetMetric({ endpoint, status: 0, durationMs });
      reportClientError({
        type: durationMs >= SLOW_THRESHOLD_MS ? "query-timeout" : "query-error",
        page,
        endpoint,
        durationMs,
        message: networkErr instanceof Error ? networkErr.message : String(networkErr),
      });
      throw networkErr;
    }

    const durationMs = Date.now() - start;
    pushNetMetric({
      endpoint,
      status: res.status,
      durationMs,
      sizeBytes: res.headers.get("content-length") ? Number(res.headers.get("content-length")) : undefined,
    });

    // Report slow loads (even if they succeed)
    if (durationMs >= SLOW_THRESHOLD_MS) {
      reportClientError({
        type: "slow-load",
        page,
        endpoint,
        durationMs,
        statusCode: res.status,
        message: `Slow response: ${endpoint} took ${durationMs}ms`,
      });
    }

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    // Report 4xx/5xx errors
    if (!res.ok) {
      reportClientError({
        type: "query-error",
        page,
        endpoint,
        durationMs,
        statusCode: res.status,
        message: `HTTP ${res.status} on ${endpoint}`,
      });
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

// Throttled invalidation: skip re-invalidating the same query key within 15s.
// Prevents request floods when phaseUpdate fires multiple times per cycle
// and multiple pages each call invalidateQueries for the same key.
const _lastInvalidated = new Map<string, number>();
export function throttledInvalidate(key: string) {
  const now = Date.now();
  const last = _lastInvalidated.get(key) ?? 0;
  if (now - last < 30_000) return;
  _lastInvalidated.set(key, now);
  queryClient.invalidateQueries({ queryKey: [key] });
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
