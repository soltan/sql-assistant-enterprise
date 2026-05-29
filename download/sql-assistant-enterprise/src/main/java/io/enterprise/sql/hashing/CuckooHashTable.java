package io.enterprise.sql.hashing;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Cuckoo Hashing implementation — O(1) worst-case lookup guarantee.
 *
 * Cuckoo Hashing is a hashing scheme that guarantees O(1) worst-case lookup
 * time, unlike standard hash tables which only offer O(1) average-case.
 * It was proposed by Pagh and Pagh (2001) and is named after the cuckoo bird
 * that kicks other eggs out of its nest.
 *
 * Algorithm:
 * - Uses TWO hash tables (T1, T2), each with its own hash function
 * - Each key can only be in ONE of TWO possible positions (one per table)
 * - Lookup: check both positions — O(1) worst case (just 2 probes)
 * - Insert: place in T1; if occupied, evict existing key to its alternate
 *   position in T2; if that's also occupied, evict that key to T1, etc.
 * - If the eviction chain exceeds MAX_DISPLACEMENTS, rehash with new seeds
 *
 * Advantages for this SQL Assistant:
 *   1. Guaranteed O(1) lookups for SimHash prefix index (SemanticHasher)
 *   2. Guaranteed O(1) lookups for LSH bucket index (LSHMultiProbe)
 *   3. Guaranteed O(1) lookups for BM25 inverted index (term → doc freq)
 *   4. Better cache behavior: no linked lists, just flat arrays
 *   5. Deterministic: same keys always produce same positions with same seeds
 *
 * Parameters:
 *   - capacity: initial capacity (will auto-expand via rehashing)
 *   - maxDisplacements: eviction chain limit before rehashing (default: 500)
 *   - loadFactor: maximum load before auto-expansion (default: 0.5)
 *
 * Thread safety: Fine-grained locking with striping (16 stripes).
 *
 * @param <K> key type
 * @param <V> value type
 */
public final class CuckooHashTable<K, V> {

    /** Maximum eviction chain length before triggering rehash */
    private static final int DEFAULT_MAX_DISPLACEMENTS = 500;

    /** Default maximum load factor before expansion */
    private static final double DEFAULT_LOAD_FACTOR = 0.5;

    /** Number of lock stripes for fine-grained concurrency */
    private static final int STRIPE_COUNT = 16;

    /** Table 1 storage */
    private volatile Entry<K, V>[] table1;

    /** Table 2 storage */
    private volatile Entry<K, V>[] table2;

    /** Hash function seeds (deterministic for reproducibility) */
    private volatile long seed1;
    private volatile long seed2;

    /** Current capacity (length of each table) */
    private volatile int capacity;

    /** Number of occupied slots across both tables */
    private final AtomicLong size;

    /** Maximum eviction chain before rehashing */
    private final int maxDisplacements;

    /** Maximum load factor before expansion */
    private final double loadFactor;

    /** Stripe locks for fine-grained concurrency */
    private final ReentrantLock[] stripes;

    /** Counter for rehash operations */
    private final AtomicLong rehashCount;

    /**
     * Entry in the hash table.
     */
    private static final class Entry<K, V> {
        final K key;
        final V value;
        final int hash1; // cached position in table1
        final int hash2; // cached position in table2

        Entry(K key, V value, int hash1, int hash2) {
            this.key = key;
            this.value = value;
            this.hash1 = hash1;
            this.hash2 = hash2;
        }
    }

    /**
     * Create a CuckooHashTable with default parameters.
     *
     * @param initialCapacity initial capacity (will be rounded up to next power of 2)
     */
    public CuckooHashTable(int initialCapacity) {
        this(initialCapacity, DEFAULT_MAX_DISPLACEMENTS, DEFAULT_LOAD_FACTOR);
    }

    /**
     * Create a CuckooHashTable with custom parameters.
     *
     * @param initialCapacity   initial capacity
     * @param maxDisplacements  max eviction chain before rehashing
     * @param loadFactor        max load factor before expansion
     */
    @SuppressWarnings("unchecked")
    public CuckooHashTable(int initialCapacity, int maxDisplacements, double loadFactor) {
        this.capacity = nextPowerOfTwo(Math.max(initialCapacity, 16));
        this.maxDisplacements = maxDisplacements;
        this.loadFactor = loadFactor;
        this.size = new AtomicLong(0);
        this.rehashCount = new AtomicLong(0);
        this.seed1 = 0x9e3779b97f4a7c15L;
        this.seed2 = 0x94d049bb133111ebL;

        this.table1 = (Entry<K, V>[]) new Entry[capacity];
        this.table2 = (Entry<K, V>[]) new Entry[capacity];

        this.stripes = new ReentrantLock[STRIPE_COUNT];
        for (int i = 0; i < STRIPE_COUNT; i++) {
            stripes[i] = new ReentrantLock();
        }
    }

    // ═══════════════════════════════════════════════════════════
    // HASH FUNCTIONS
    // ═══════════════════════════════════════════════════════════

    /**
     * Primary hash function (MurmurHash3-like).
     */
    private int hash1(K key) {
        long h = seed1;
        h ^= key.hashCode();
        h *= 0x5bd1e9959e8c94c6L;
        h ^= h >>> 47;
        h *= 0x5bd1e9959e8c94c6L;
        h ^= h >>> 47;
        return (int) (Math.abs(h) % capacity);
    }

    /**
     * Secondary hash function (different seed).
     */
    private int hash2(K key) {
        long h = seed2;
        h ^= key.hashCode();
        h *= 0x5bd1e9959e8c94c6L;
        h ^= h >>> 47;
        h *= 0x5bd1e9959e8c94c6L;
        h ^= h >>> 47;
        return (int) (Math.abs(h) % capacity);
    }

    /**
     * Recompute hash positions after a rehash (capacity change or seed change).
     */
    private int[] computeHashes(K key) {
        return new int[] { hash1(key), hash2(key) };
    }

    // ═══════════════════════════════════════════════════════════
    // CORE OPERATIONS — O(1) worst case
    // ═══════════════════════════════════════════════════════════

    /**
     * Get the value for a key — O(1) WORST CASE.
     * Checks only two possible positions (one per table).
     *
     * @param key key to look up
     * @return value, or null if not found
     */
    public V get(K key) {
        if (key == null) return null;

        int h1 = hash1(key);
        int h2 = hash2(key);

        // Check table1 — no lock needed for reads (volatile arrays)
        Entry<K, V> e1 = table1[h1];
        if (e1 != null && key.equals(e1.key)) {
            return e1.value;
        }

        // Check table2
        Entry<K, V> e2 = table2[h2];
        if (e2 != null && key.equals(e2.key)) {
            return e2.value;
        }

        return null;
    }

    /**
     * Check if a key exists — O(1) WORST CASE.
     *
     * @param key key to check
     * @return true if key is present
     */
    public boolean containsKey(K key) {
        return get(key) != null;
    }

    /**
     * Put a key-value pair — O(1) amortized (O(n) on rehash, but amortized O(1)).
     * If the key already exists, its value is updated.
     *
     * @param key   key
     * @param value value
     * @return previous value, or null
     */
    public V put(K key, V value) {
        if (key == null) throw new NullPointerException("Key cannot be null");

        int stripeIdx = Math.abs(key.hashCode()) % STRIPE_COUNT;
        stripes[stripeIdx].lock();
        try {
            // First, check if key already exists (update)
            int h1 = hash1(key);
            int h2 = hash2(key);

            Entry<K, V> e1 = table1[h1];
            if (e1 != null && key.equals(e1.key)) {
                V old = e1.value;
                table1[h1] = new Entry<>(key, value, h1, h2);
                return old;
            }

            Entry<K, V> e2 = table2[h2];
            if (e2 != null && key.equals(e2.key)) {
                V old = e2.value;
                table2[h2] = new Entry<>(key, value, h1, h2);
                return old;
            }

            // Key doesn't exist — insert via cuckoo eviction
            return insertNew(key, value, h1, h2);
        } finally {
            stripes[stripeIdx].unlock();
        }
    }

    /**
     * Insert a new key using the cuckoo eviction strategy.
     * If eviction chain exceeds maxDisplacements, triggers a rehash.
     */
    private V insertNew(K key, V value, int h1, int h2) {
        // Try table1 first
        if (table1[h1] == null) {
            table1[h1] = new Entry<>(key, value, h1, h2);
            size.incrementAndGet();
            checkLoadFactor();
            return null;
        }

        // Try table2
        if (table2[h2] == null) {
            table2[h2] = new Entry<>(key, value, h1, h2);
            size.incrementAndGet();
            checkLoadFactor();
            return null;
        }

        // Both positions occupied — start cuckoo eviction chain
        K currentKey = key;
        V currentValue = value;
        int currentH1 = h1;
        int currentH2 = h2;
        boolean useTable1 = true; // start by placing in table1

        for (int i = 0; i < maxDisplacements; i++) {
            if (useTable1) {
                // Evict entry at table1[currentH1]
                Entry<K, V> evicted = table1[currentH1];
                table1[currentH1] = new Entry<>(currentKey, currentValue, currentH1, currentH2);

                // The evicted key goes to its alternate position in table2
                currentKey = evicted.key;
                currentValue = evicted.value;
                currentH1 = evicted.hash1;
                currentH2 = evicted.hash2;

                // Check if table2 position is free
                if (table2[currentH2] == null) {
                    table2[currentH2] = new Entry<>(currentKey, currentValue, currentH1, currentH2);
                    size.incrementAndGet();
                    checkLoadFactor();
                    return null;
                }
            } else {
                // Evict entry at table2[currentH2]
                Entry<K, V> evicted = table2[currentH2];
                table2[currentH2] = new Entry<>(currentKey, currentValue, currentH1, currentH2);

                // The evicted key goes to its alternate position in table1
                currentKey = evicted.key;
                currentValue = evicted.value;
                currentH1 = evicted.hash1;
                currentH2 = evicted.hash2;

                // Check if table1 position is free
                if (table1[currentH1] == null) {
                    table1[currentH1] = new Entry<>(currentKey, currentValue, currentH1, currentH2);
                    size.incrementAndGet();
                    checkLoadFactor();
                    return null;
                }
            }

            useTable1 = !useTable1;
        }

        // Max displacements exceeded — rehash with new seeds and larger capacity
        rehash();
        // Retry insertion after rehash
        return put(currentKey, currentValue);
    }

    /**
     * Remove a key — O(1) WORST CASE.
     *
     * @param key key to remove
     * @return previous value, or null if not found
     */
    public V remove(K key) {
        if (key == null) return null;

        int stripeIdx = Math.abs(key.hashCode()) % STRIPE_COUNT;
        stripes[stripeIdx].lock();
        try {
            int h1 = hash1(key);
            int h2 = hash2(key);

            Entry<K, V> e1 = table1[h1];
            if (e1 != null && key.equals(e1.key)) {
                table1[h1] = null;
                size.decrementAndGet();
                return e1.value;
            }

            Entry<K, V> e2 = table2[h2];
            if (e2 != null && key.equals(e2.key)) {
                table2[h2] = null;
                size.decrementAndGet();
                return e2.value;
            }

            return null;
        } finally {
            stripes[stripeIdx].unlock();
        }
    }

    // ═══════════════════════════════════════════════════════════
    // REHASHING & EXPANSION
    // ═══════════════════════════════════════════════════════════

    /**
     * Check if load factor threshold is exceeded and trigger expansion.
     */
    private void checkLoadFactor() {
        double currentLoad = (double) size.get() / (2.0 * capacity);
        if (currentLoad > loadFactor) {
            rehash();
        }
    }

    /**
     * Rehash with new seeds and expanded capacity.
     * This is triggered when:
     *   1. Load factor exceeds threshold
     *   2. Eviction chain exceeds maxDisplacements
     *
     * Double the capacity and generate new hash seeds.
     */
    @SuppressWarnings("unchecked")
    private void rehash() {
        // Acquire all stripes for global rehash
        for (ReentrantLock lock : stripes) {
            lock.lock();
        }

        try {
            // Collect all existing entries
            List<Entry<K, V>> allEntries = new ArrayList<>();
            for (int i = 0; i < capacity; i++) {
                if (table1[i] != null) allEntries.add(table1[i]);
                if (table2[i] != null) allEntries.add(table2[i]);
            }

            // Double capacity and change seeds
            int newCapacity = capacity * 2;
            long newSeed1 = seed1 ^ 0x5bd1e9959e8c94c6L;
            long newSeed2 = seed2 ^ 0x94d049bb133111ebL;

            // Swap to new tables
            Entry<K, V>[] newTable1 = (Entry<K, V>[]) new Entry[newCapacity];
            Entry<K, V>[] newTable2 = (Entry<K, V>[]) new Entry[newCapacity];

            this.capacity = newCapacity;
            this.seed1 = newSeed1;
            this.seed2 = newSeed2;
            this.table1 = newTable1;
            this.table2 = newTable2;

            // Re-insert all entries
            for (Entry<K, V> entry : allEntries) {
                int h1 = hash1(entry.key);
                int h2 = hash2(entry.key);
                insertDirect(entry.key, entry.value, h1, h2, maxDisplacements);
            }

            rehashCount.incrementAndGet();
        } finally {
            for (int i = STRIPE_COUNT - 1; i >= 0; i--) {
                stripes[i].unlock();
            }
        }
    }

    /**
     * Direct insertion for rehash (no load factor checks, no locks).
     */
    private void insertDirect(K key, V value, int h1, int h2, int maxTries) {
        K currentKey = key;
        V currentValue = value;
        int currentH1 = h1;
        int currentH2 = h2;
        boolean useTable1 = true;

        for (int i = 0; i < maxTries; i++) {
            if (useTable1) {
                if (table1[currentH1] == null) {
                    table1[currentH1] = new Entry<>(currentKey, currentValue, currentH1, currentH2);
                    return;
                }
                Entry<K, V> evicted = table1[currentH1];
                table1[currentH1] = new Entry<>(currentKey, currentValue, currentH1, currentH2);
                currentKey = evicted.key;
                currentValue = evicted.value;
                currentH1 = evicted.hash1;
                currentH2 = evicted.hash2;
            } else {
                if (table2[currentH2] == null) {
                    table2[currentH2] = new Entry<>(currentKey, currentValue, currentH1, currentH2);
                    return;
                }
                Entry<K, V> evicted = table2[currentH2];
                table2[currentH2] = new Entry<>(currentKey, currentValue, currentH1, currentH2);
                currentKey = evicted.key;
                currentValue = evicted.value;
                currentH1 = evicted.hash1;
                currentH2 = evicted.hash2;
            }
            useTable1 = !useTable1;
        }
        // Shouldn't happen with expanded capacity, but fallback
        throw new IllegalStateException("CuckooHashTable rehash failed after expansion to " + capacity);
    }

    // ═══════════════════════════════════════════════════════════
    // BULK OPERATIONS
    // ═══════════════════════════════════════════════════════════

    /**
     * Put all entries from a map.
     *
     * @param map entries to add
     */
    public void putAll(Map<? extends K, ? extends V> map) {
        for (Map.Entry<? extends K, ? extends V> entry : map.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Get or compute a value — atomic get-if-absent-then-compute.
     *
     * @param key      key
     * @param computer function to compute value if key is absent
     * @return existing or newly computed value
     */
    public V computeIfAbsent(K key, java.util.function.Function<K, V> computer) {
        V existing = get(key);
        if (existing != null) return existing;

        int stripeIdx = Math.abs(key.hashCode()) % STRIPE_COUNT;
        stripes[stripeIdx].lock();
        try {
            // Double-check under lock
            V v = get(key);
            if (v != null) return v;

            V computed = computer.apply(key);
            if (computed != null) {
                put(key, computed);
            }
            return computed;
        } finally {
            stripes[stripeIdx].unlock();
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ITERATION & VIEWS
    // ═══════════════════════════════════════════════════════════

    /**
     * Get all keys.
     *
     * @return set of all keys
     */
    public Set<K> keySet() {
        Set<K> keys = new LinkedHashSet<>();
        for (int i = 0; i < capacity; i++) {
            if (table1[i] != null) keys.add(table1[i].key);
            if (table2[i] != null) keys.add(table2[i].key);
        }
        return keys;
    }

    /**
     * Get all values.
     *
     * @return collection of all values
     */
    public Collection<V> values() {
        List<V> vals = new ArrayList<>();
        for (int i = 0; i < capacity; i++) {
            if (table1[i] != null) vals.add(table1[i].value);
            if (table2[i] != null) vals.add(table2[i].value);
        }
        return vals;
    }

    /**
     * Get all entries.
     *
     * @return set of all key-value entries
     */
    public Set<Map.Entry<K, V>> entrySet() {
        Set<Map.Entry<K, V>> entries = new LinkedHashSet<>();
        for (int i = 0; i < capacity; i++) {
            if (table1[i] != null) entries.add(Map.entry(table1[i].key, table1[i].value));
            if (table2[i] != null) entries.add(Map.entry(table2[i].key, table2[i].value));
        }
        return entries;
    }

    /**
     * Iterate over all entries with a consumer.
     *
     * @param consumer action to apply to each entry
     */
    public void forEach(java.util.function.BiConsumer<K, V> consumer) {
        for (int i = 0; i < capacity; i++) {
            Entry<K, V> e1 = table1[i];
            if (e1 != null) consumer.accept(e1.key, e1.value);
            Entry<K, V> e2 = table2[i];
            if (e2 != null) consumer.accept(e2.key, e2.value);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // STATISTICS & ACCESSORS
    // ═══════════════════════════════════════════════════════════

    /** Get number of entries */
    public long size() { return size.get(); }

    /** Check if empty */
    public boolean isEmpty() { return size.get() == 0; }

    /** Get current capacity */
    public int capacity() { return capacity; }

    /** Get number of rehash operations performed */
    public long rehashCount() { return rehashCount.get(); }

    /** Get current load factor */
    public double loadFactor() { return (double) size.get() / (2.0 * capacity); }

    /** Get statistics map */
    public Map<String, Object> stats() {
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("size", size.get());
        s.put("capacity", capacity);
        s.put("loadFactor", String.format("%.3f", loadFactor()));
        s.put("rehashCount", rehashCount.get());
        s.put("maxDisplacements", maxDisplacements);

        // Count actual occupancy per table
        int t1Occupied = 0, t2Occupied = 0;
        for (int i = 0; i < capacity; i++) {
            if (table1[i] != null) t1Occupied++;
            if (table2[i] != null) t2Occupied++;
        }
        s.put("table1Occupancy", t1Occupied);
        s.put("table2Occupancy", t2Occupied);

        return s;
    }

    @Override
    public String toString() {
        return String.format("CuckooHashTable{size=%d, capacity=%d, load=%.2f%%, rehashes=%d}",
            size.get(), capacity, loadFactor() * 100, rehashCount.get());
    }

    // ═══════════════════════════════════════════════════════════
    // UTILITIES
    // ═══════════════════════════════════════════════════════════

    /**
     * Find the next power of 2 >= n.
     */
    private static int nextPowerOfTwo(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    /**
     * Clear all entries.
     */
    @SuppressWarnings("unchecked")
    public void clear() {
        for (ReentrantLock lock : stripes) {
            lock.lock();
        }
        try {
            table1 = (Entry<K, V>[]) new Entry[capacity];
            table2 = (Entry<K, V>[]) new Entry[capacity];
            size.set(0);
        } finally {
            for (int i = STRIPE_COUNT - 1; i >= 0; i--) {
                stripes[i].unlock();
            }
        }
    }
}
