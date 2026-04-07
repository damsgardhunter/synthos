/**
 * Shared OpenAI circuit breaker.
 *
 * All modules that call OpenAI import from here so that a timeout in
 * data-fetcher.ts immediately protects synthesis-tracker.ts (and vice-versa).
 * Previously each file had its own independent AI_CIRCUIT, so one module
 * could have its circuit open while another kept burning 20s timeouts.
 *
 * Thresholds:
 *   FAIL_THRESHOLD = 2  — trip after 2 consecutive timeouts (not 3)
 *   BACKOFF_MS = 20 min — long enough that a brief OpenAI outage doesn't
 *                         spam the feed every 5 min with fresh timeout clusters
 */

const FAIL_THRESHOLD = 2;
const BACKOFF_MS = 20 * 60 * 1000; // 20 minutes

const circuit = { fails: 0, backoffUntil: 0 };

export function openaiCircuitOpen(): boolean {
  return Date.now() < circuit.backoffUntil;
}

export function recordOpenAIFail(): void {
  circuit.fails++;
  if (circuit.fails >= FAIL_THRESHOLD) {
    circuit.backoffUntil = Date.now() + BACKOFF_MS;
    circuit.fails = 0;
    const until = new Date(circuit.backoffUntil).toISOString();
    console.warn(`[OpenAI-Circuit] Opened after ${FAIL_THRESHOLD} consecutive failures — backing off until ${until}`);
  }
}

export function recordOpenAISuccess(): void {
  circuit.fails = 0;
  circuit.backoffUntil = 0;
}

/** Returns ms until circuit closes, or 0 if already closed. */
export function openaiCircuitBackoffMs(): number {
  return Math.max(0, circuit.backoffUntil - Date.now());
}
