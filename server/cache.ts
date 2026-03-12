interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

class MemoryCache {
  private store = new Map<string, CacheEntry<any>>();

  get<T>(key: string): T | undefined {
    const entry = this.store.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return undefined;
    }
    return entry.data as T;
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
    const data = await fetcher();
    this.set(key, data, ttlMs);
    return data;
  }

  clear(): void {
    this.store.clear();
  }
}

export const cache = new MemoryCache();

export const TTL = {
  ELEMENTS: 60 * 60 * 1000,
  STATS: 30 * 1000,
  CRYSTAL_STRUCTURES_BY_FORMULA: 5 * 60 * 1000,
  COMPUTATIONAL_RESULTS_BY_FORMULA: 5 * 60 * 1000,
  RESEARCH_LOGS: 15 * 1000,
  LEARNING_PHASES: 30 * 1000,
  DFT_STATUS: 30 * 1000,
  ENGINE_MEMORY: 30 * 1000,
  CANDIDATES: 20 * 1000,
  STRATEGY: 30 * 1000,
  MILESTONES: 60 * 1000,
  NOVEL_INSIGHTS: 30 * 1000,
  CONVERGENCE: 30 * 1000,
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
  CONVERGENCE: "convergence",
  crystalStructuresByFormula: (formula: string) => `crystal-structures:${formula}`,
  computationalResultsByFormula: (formula: string) => `computational-results:${formula}`,
};
