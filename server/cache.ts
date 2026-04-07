interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

const MAX_CACHE_ENTRIES = 500;

class MemoryCache {
  private store = new Map<string, CacheEntry<any>>();
  private inflight = new Map<string, Promise<any>>();
  private sweepTimer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // Sweep expired entries every 5 minutes so they don't accumulate
    // indefinitely for keys that are never read again.
    this.sweepTimer = setInterval(() => this.sweepExpired(), 5 * 60 * 1000);
    if (this.sweepTimer?.unref) this.sweepTimer.unref(); // don't block process exit
  }

  private sweepExpired(): void {
    const now = Date.now();
    for (const [key, entry] of this.store) {
      if (now > entry.expiresAt) this.store.delete(key);
    }
    // Hard cap: if still over limit after sweep, evict oldest-expiring entries
    if (this.store.size > MAX_CACHE_ENTRIES) {
      const sorted = [...this.store.entries()].sort((a, b) => a[1].expiresAt - b[1].expiresAt);
      for (const [key] of sorted.slice(0, this.store.size - MAX_CACHE_ENTRIES)) {
        this.store.delete(key);
      }
    }
  }

  get<T>(key: string): T | undefined {
    const entry = this.store.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return undefined;
    }
    return entry.data as T;
  }

  getStale<T>(key: string): T | undefined {
    return (this.store.get(key)?.data as T) ?? undefined;
  }

  isExpired(key: string): boolean {
    const entry = this.store.get(key);
    if (!entry) return true;
    return Date.now() > entry.expiresAt;
  }

  set<T>(key: string, data: T, ttlMs: number): void {
    this.store.set(key, { data, expiresAt: Date.now() + ttlMs });
  }

  invalidate(key: string): void {
    this.store.delete(key);
  }

  invalidatePrefix(prefix: string): void {
    const keysToDelete: string[] = [];
    this.store.forEach((_val, key) => {
      if (key.startsWith(prefix)) {
        keysToDelete.push(key);
      }
    });
    keysToDelete.forEach(key => this.store.delete(key));
  }

  async getOrSet<T>(key: string, ttlMs: number, fetcher: () => Promise<T>): Promise<T> {
    const cached = this.get<T>(key);
    if (cached !== undefined) return cached;

    const existing = this.inflight.get(key);
    if (existing) return existing as Promise<T>;

    const promise = fetcher().then(data => {
      this.set(key, data, ttlMs);
      this.inflight.delete(key);
      return data;
    }).catch(err => {
      this.inflight.delete(key);
      throw err;
    });

    this.inflight.set(key, promise);
    return promise;
  }

  // Returns stale data immediately if available while refreshing in background.
  // Only blocks on the very first ever fetch (no stale data exists yet).
  async getOrSetStale<T>(key: string, ttlMs: number, fetcher: () => Promise<T>): Promise<T> {
    const stale = this.getStale<T>(key);
    const expired = this.isExpired(key);

    if (stale !== undefined && !expired) return stale;

    if (stale !== undefined && expired) {
      // Serve stale immediately; refresh in background (deduplicated)
      if (!this.inflight.has(key)) {
        const promise = fetcher().then(data => {
          this.set(key, data, ttlMs);
          this.inflight.delete(key);
          return data;
        }).catch(() => { this.inflight.delete(key); });
        this.inflight.set(key, promise);
      }
      return stale;
    }

    return this.getOrSet(key, ttlMs, fetcher);
  }

  clear(): void {
    this.store.clear();
  }
}

export const cache = new MemoryCache();

export const TTL = {
  ELEMENTS: 60 * 60 * 1000,
  STATS: 10 * 60 * 1000,  // getStats() makes 3 serial DB round-trips — too slow to re-run every 3 min under GB training load
  CRYSTAL_STRUCTURES_BY_FORMULA: 5 * 60 * 1000,
  COMPUTATIONAL_RESULTS_BY_FORMULA: 5 * 60 * 1000,
  RESEARCH_LOGS: 60 * 1000,
  LEARNING_PHASES: 3 * 60 * 1000,
  DFT_STATUS: 2 * 60 * 1000,
  ENGINE_MEMORY: 5 * 60 * 1000,
  CANDIDATES: 2 * 60 * 1000,
  STRATEGY: 3 * 60 * 1000,
  MILESTONES: 5 * 60 * 1000,
  NOVEL_INSIGHTS: 20 * 60 * 1000,
  THEORY_REPORT: 20 * 60 * 1000,
  CONVERGENCE: 2 * 60 * 1000,
  TSC_CANDIDATES: 2 * 60 * 1000,  // 500-candidate scan + sort — cache for 2 min
  UNIFIED_CI: 15 * 60 * 1000,    // ML+GNN inference result — expensive, cache for 15 min
  NOVEL_PREDICTIONS: 2 * 60 * 1000, // novel prediction rows — cache for 2 min
  PARETO_FRONTIER: 2 * 60 * 1000,  // unused — route serves latestParetoResults directly
};

export const CACHE_KEYS = {
  ELEMENTS: "elements:all",
  STATS: "stats:all",
  RESEARCH_LOGS: "research-logs",
  LEARNING_PHASES: "learning-phases",
  DFT_STATUS: "dft-status",
  ENGINE_MEMORY: "engine-memory",
  CANDIDATES: "candidates",
  CANDIDATES_BY_TC: "candidates-by-tc",
  STRATEGY_LATEST: "strategy-latest",
  STRATEGY_HISTORY: "strategy-history",
  MILESTONES: "milestones",
  NOVEL_INSIGHTS: "novel-insights",
  THEORY_REPORT: "theory-report",
  CONVERGENCE: "convergence",
  TSC_CANDIDATES: "tsc-candidates",
  NOVEL_PREDICTIONS: "novel-predictions",
  PARETO_FRONTIER: "pareto-frontier",
  crystalStructuresByFormula: (formula: string) => `crystal-structures:${formula}`,
  computationalResultsByFormula: (formula: string) => `computational-results:${formula}`,
  unifiedCI: (formula: string) => `unified-ci:${formula}`,
};
