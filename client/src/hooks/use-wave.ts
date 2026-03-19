import { useState, useEffect } from "react";

/**
 * Returns false for the first `delayMs` milliseconds after the module loads,
 * then returns true. Use as the `enabled` prop on non-critical useQuery calls
 * to stagger the initial burst of requests and avoid saturating the DB pool.
 *
 * All calls with the same delayMs share the same timer (module-level) so the
 * wave fires once across the whole app, not once per component mount.
 */
const timers = new Map<number, { ready: boolean; callbacks: Set<() => void> }>();

function getWave(delayMs: number) {
  if (!timers.has(delayMs)) {
    const entry = { ready: false, callbacks: new Set<() => void>() };
    timers.set(delayMs, entry);
    setTimeout(() => {
      entry.ready = true;
      entry.callbacks.forEach((cb) => cb());
    }, delayMs);
  }
  return timers.get(delayMs)!;
}

export function useWave(delayMs: number): boolean {
  const wave = getWave(delayMs);
  const [ready, setReady] = useState(() => wave.ready);

  useEffect(() => {
    if (wave.ready) return;
    const cb = () => setReady(true);
    wave.callbacks.add(cb);
    return () => { wave.callbacks.delete(cb); };
  }, [wave]);

  return ready;
}
