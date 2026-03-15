interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

class MemoryCache {
  private store = new Map<string, CacheEntry<any>>();
  private inflight = new Map<string, Promise<any>>();

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
  STATS: 3 * 60 * 1000,          // was 30s — DB hit every 30s was causing 1.1s delays
  CRYSTAL_STRUCTURES_BY_FORMULA: 5 * 60 * 1000,
  COMPUTATIONAL_RESULTS_BY_FORMULA: 5 * 60 * 1000,
  RESEARCH_LOGS: 60 * 1000,      // was 15s
  LEARNING_PHASES: 3 * 60 * 1000, // was 30s
  DFT_STATUS: 2 * 60 * 1000,     // was 30s
  ENGINE_MEMORY: 5 * 60 * 1000,
  CANDIDATES: 2 * 60 * 1000,     // was 20s
  STRATEGY: 3 * 60 * 1000,       // was 30s
  MILESTONES: 5 * 60 * 1000,     // was 60s
  NOVEL_INSIGHTS: 20 * 60 * 1000, // 20min — data only changes when ENABLE_INSIGHT_NOVELTY=true
  THEORY_REPORT: 20 * 60 * 1000,  // 20min — in-memory causal/symbolic stats, changes slowly
  CONVERGENCE: 2 * 60 * 1000,    // was 30s
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
  crystalStructuresByFormula: (formula: string) => `crystal-structures:${formula}`,
  computationalResultsByFormula: (formula: string) => `computational-results:${formula}`,
};
