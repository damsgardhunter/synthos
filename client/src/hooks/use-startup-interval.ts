import { useState, useEffect } from "react";

// Module-level timestamp set once when JS loads. Shared across all pages
// so they all resume polling at exactly the same time.
export const STARTUP_UNTIL = Date.now() + 5 * 60 * 1000;

/**
 * Returns `false` (no auto-poll) for the first 5 minutes after page load,
 * then returns `normalMs` so polling resumes once the server has settled.
 */
export function useStartupSafeInterval(normalMs: number): number | false {
  const [past, setPast] = useState(() => Date.now() >= STARTUP_UNTIL);
  useEffect(() => {
    if (past) return;
    const remaining = STARTUP_UNTIL - Date.now();
    if (remaining <= 0) { setPast(true); return; }
    const t = setTimeout(() => setPast(true), remaining);
    return () => clearTimeout(t);
  }, [past]);
  return past ? normalMs : false;
}
