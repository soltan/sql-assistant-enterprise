package io.enterprise.sql.hashing;

import io.enterprise.sql.model.SemanticHash;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * LSH Multi-Probe Index — dramatically improves recall over single-hash SimHash.
 * 
 * Problem with single SimHash: nearby items may differ in a few bits that cross
 * the 16-bit prefix boundary, causing them to land in different buckets → missed matches.
 * 
 * Solution: Multiple hash tables with DIFFERENT hash functions (different seeds).
 * Each table partitions the hash space differently. A query is probed in ALL tables,
 * and results are unioned. This gives near-guaranteed recall for items within
 * Hamming distance k as long as num_tables * (1 - k/128) is large enough.
 * 
 * Additionally implements "multi-probe": for each table, we also check
 * neighboring buckets (bit flips at positions with lowest weight) to recover
 * items that just barely missed the exact bucket.
 *
 * Parameters:
 * - numTables: number of independent hash tables (typical: 5-12)
 * - numProbes: number of additional neighboring buckets to check per table (typical: 3-10)
 * - bucketWidth: number of prefix bits for bucketing (typical: 16)
 *
 * Time complexity: O(numTables * numProbes) per lookup
 * Space complexity: O(numTables * n) where n = number of indexed items
 * Deterministic: all hash functions use fixed seeds
 */
public class LSHMultiProbe {

    /** Number of independent hash tables */
    private final int numTables;

    /** Number of additional neighbor probes per table */
    private final int numProbes;

    /** Number of prefix bits for bucketing */
    private final int bucketBits;

    /** Hash functions: each table uses a different seed */
    private final long[] hashSeeds;

    /** Bucket index per table: prefix → Set of entry IDs */
    private final List<ConcurrentHashMap<Long, Set<IndexEntry>>> tables;

    /** All indexed entries */
    private final ConcurrentHashMap<Long, IndexEntry> entries;

    /** Entry record */
    public record IndexEntry(long id, SemanticHash originalHash, String intentType, String exampleQuery) {}

    /**
     * Create an LSH Multi-Probe index.
     */
    public LSHMultiProbe(int numTables, int numProbes, int bucketBits) {
        this.numTables = numTables;
        this.numProbes = numProbes;
        this.bucketBits = bucketBits;
        this.hashSeeds = new long[numTables];
        this.tables = new ArrayList<>(numTables);
        this.entries = new ConcurrentHashMap<>();

        // Generate deterministic seeds for each table
        Random rng = new Random(0x9e3779b97f4a7c15L);
        for (int i = 0; i < numTables; i++) {
            hashSeeds[i] = rng.nextLong();
            tables.add(new ConcurrentHashMap<>());
        }
    }

    /**
     * Compute a perturbed hash for a given table by XOR-ing with the table seed.
     * This effectively rotates the hash space so that different tables have
     * different bucket boundaries.
     */
    private SemanticHash perturbHash(SemanticHash original, int tableIndex) {
        long seed = hashSeeds[tableIndex];
        // XOR with seed-based rotation to create different bucket partition
        long high = original.high() ^ (seed ^ (seed >>> 32));
        long low = original.low() ^ (seed << 32 ^ seed);
        return new SemanticHash(high, low);
    }

    /**
     * Extract the bucket prefix from a hash (top N bits of high word).
     */
    private long bucketOf(SemanticHash hash) {
        return hash.high() >>> (64 - bucketBits);
    }

    /**
     * Generate neighbor bucket keys by flipping the lowest-weight bits.
     * This is the "multi-probe" part: we check buckets that are 1-2 bit flips
     * away from the query bucket, catching items that barely missed.
     */
    private List<Long> probeBuckets(SemanticHash hash) {
        long baseBucket = bucketOf(hash);
        List<Long> buckets = new ArrayList<>();
        buckets.add(baseBucket);

        // Generate flips at positions within bucketBits range
        // Lower bit positions have less impact on the hash → probe them first
        for (int flip = 0; flip < numProbes && flip < bucketBits; flip++) {
            long flipped = baseBucket ^ (1L << flip);
            buckets.add(flipped);
        }

        // Second-order flips (2 bits) for more coverage
        if (numProbes > bucketBits) {
            for (int i = 0; i < bucketBits && buckets.size() < numProbes + 1; i++) {
                for (int j = i + 1; j < bucketBits && buckets.size() < numProbes + 1; j++) {
                    long flipped = baseBucket ^ (1L << i) ^ (1L << j);
                    buckets.add(flipped);
                }
            }
        }

        return buckets;
    }

    /**
     * Index an entry across all hash tables.
     */
    public void index(long id, SemanticHash hash, String intentType, String exampleQuery) {
        IndexEntry entry = new IndexEntry(id, hash, intentType, exampleQuery);
        entries.put(id, entry);

        for (int t = 0; t < numTables; t++) {
            SemanticHash perturbed = perturbHash(hash, t);
            long bucket = bucketOf(perturbed);
            tables.get(t)
                .computeIfAbsent(bucket, k -> ConcurrentHashMap.newKeySet())
                .add(entry);
        }
    }

    /**
     * Query the index: find all entries within maxDistance Hamming distance.
     * Uses multi-table lookup + multi-probe for maximum recall.
     * Re-ranks results by actual Hamming distance.
     */
    public List<MatchResult> query(SemanticHash queryHash, int maxDistance, int topK) {
        // Collect candidates from all tables with multi-probe
        Set<IndexEntry> candidates = new LinkedHashSet<>();

        for (int t = 0; t < numTables; t++) {
            SemanticHash perturbed = perturbHash(queryHash, t);
            List<Long> buckets = probeBuckets(perturbed);

            for (long bucket : buckets) {
                Set<IndexEntry> bucketEntries = tables.get(t).get(bucket);
                if (bucketEntries != null) {
                    candidates.addAll(bucketEntries);
                }
            }
        }

        // Re-rank by actual Hamming distance
        record Scored(IndexEntry entry, double similarity) {}
        List<Scored> scored = new ArrayList<>();

        for (IndexEntry candidate : candidates) {
            double sim = queryHash.similarity(candidate.originalHash());
            if (sim >= (1.0 - (double) maxDistance / 128.0)) {
                scored.add(new Scored(candidate, sim));
            }
        }

        scored.sort(Comparator.comparingDouble(Scored::similarity).reversed());

        return scored.stream()
            .limit(topK)
            .map(s -> new MatchResult(
                s.entry().id(),
                s.entry().originalHash(),
                s.entry().intentType(),
                s.similarity()
            ))
            .toList();
    }

    /**
     * Get the number of indexed entries.
     */
    public int size() {
        return entries.size();
    }

    /**
     * Get statistics about the index.
     */
    public Map<String, Object> stats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("entries", entries.size());
        stats.put("numTables", numTables);
        stats.put("numProbes", numProbes);
        stats.put("bucketBits", bucketBits);
        
        long totalBuckets = tables.stream()
            .mapToLong(t -> t.size())
            .sum();
        stats.put("totalBuckets", totalBuckets);
        
        return stats;
    }

    /**
     * Match result record.
     */
    public record MatchResult(long id, SemanticHash hash, String intentType, double similarity) {}

    /**
     * Builder for LSHMultiProbe with sensible defaults.
     */
    public static class Builder {
        private int numTables = 8;
        private int numProbes = 5;
        private int bucketBits = 16;

        public Builder numTables(int val) { this.numTables = val; return this; }
        public Builder numProbes(int val) { this.numProbes = val; return this; }
        public Builder bucketBits(int val) { this.bucketBits = val; return this; }

        public LSHMultiProbe build() {
            return new LSHMultiProbe(numTables, numProbes, bucketBits);
        }
    }
}
