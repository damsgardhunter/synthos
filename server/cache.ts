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
};

export const CACHE_KEYS = {
  ELEMENTS: "elements:all",
  STATS: "stats:all",
  crystalStructuresByFormula: (formula: string) => `crystal-structures:${formula}`,
  computationalResultsByFormula: (formula: string) => `computational-results:${formula}`,
};
