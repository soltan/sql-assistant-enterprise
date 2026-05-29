package io.enterprise.sql.hashing;

import io.enterprise.sql.model.SemanticHash;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Semantic hasher that combines SimHash with intent-specific fingerprinting.
 * Provides a two-tier hashing strategy:
 *
 * Tier 1: SimHash for fast approximate matching (Hamming distance filtering)
 * Tier 2: Intent fingerprint for exact intent classification (keyword patterns)
 *
 * CuckooHashTable Integration:
 *   - Prefix index now uses CuckooHashTable for O(1) worst-case lookups
 *   - Previously: ConcurrentHashMap<Long, List<HashEntry>> → O(1) average
 *   - Now: CuckooHashTable<Long, List<HashEntry>> → O(1) guaranteed worst case
 *   - Critical for high-throughput scenarios where hash collisions could cause
 *     O(n) degradation in standard hash tables
 *
 * Also maintains a hash cluster index for O(1) approximate lookup.
 */
public class SemanticHasher {

    /** Cluster radius in Hamming distance for grouping similar hashes */
    private static final int CLUSTER_RADIUS = 8;

    /** Inverted index: hash prefix → cluster of known hashes — CuckooHashTable for O(1) worst-case */
    private final CuckooHashTable<Long, List<HashEntry>> prefixIndex;

    /** Known intent hashes for lookup — CuckooHashTable for O(1) worst-case */
    private final CuckooHashTable<SemanticHash, String> hashToIntentType;

    public SemanticHasher() {
        this.prefixIndex = new CuckooHashTable<>(64);
        this.hashToIntentType = new CuckooHashTable<>(64);
    }

    /**
     * Register a known intent pattern.
     *
     * @param exampleQuery example query that maps to this intent
     * @param intentType   the SqlIntent type name
     */
    public void registerPattern(String exampleQuery, String intentType) {
        SemanticHash hash = SimHash.computeFromQuery(exampleQuery);
        hashToIntentType.put(hash, intentType);

        // Index by 16-bit prefix of high word for fast filtering
        // O(1) worst-case via CuckooHashTable
        long prefix = hash.high() >>> 48;
        List<HashEntry> existing = prefixIndex.get(prefix);
        if (existing == null) {
            List<HashEntry> newList = new CopyOnWriteArrayList<>();
            newList.add(new HashEntry(hash, intentType, exampleQuery));
            prefixIndex.put(prefix, newList);
        } else {
            ((CopyOnWriteArrayList<HashEntry>) existing).add(new HashEntry(hash, intentType, exampleQuery));
        }
    }

    /**
     * Hash a query and find the best matching intent type.
     * Uses CuckooHashTable for O(1) worst-case prefix lookups.
     *
     * @param query user query
     * @return match result with confidence
     */
    public MatchResult match(String query) {
        SemanticHash queryHash = SimHash.computeFromQuery(query);

        // Phase 1: Prefix filtering — only check clusters with same 16-bit prefix
        // O(1) worst-case lookup via CuckooHashTable
        long prefix = queryHash.high() >>> 48;
        List<HashEntry> candidates = prefixIndex.getOrDefault(prefix, List.of());

        // Also check neighboring prefixes (±1 for tolerance)
        candidates = new ArrayList<>(candidates);
        List<HashEntry> left = prefixIndex.get(prefix - 1);
        List<HashEntry> right = prefixIndex.get(prefix + 1);
        if (left != null) candidates.addAll(left);
        if (right != null) candidates.addAll(right);

        if (candidates.isEmpty()) {
            return new MatchResult(queryHash, "Unknown", 0.0);
        }

        // Phase 2: Hamming distance ranking
        MatchResult best = null;
        for (HashEntry entry : candidates) {
            double similarity = queryHash.similarity(entry.hash);

            if (best == null || similarity > best.confidence()) {
                best = new MatchResult(queryHash, entry.intentType, similarity);
            }
        }

        return best;
    }

    /**
     * Hash a query string into its semantic hash.
     */
    public SemanticHash hash(String query) {
        return SimHash.computeFromQuery(query);
    }

    /**
     * Number of registered patterns.
     */
    public int patternCount() {
        return (int) hashToIntentType.size();
    }

    /**
     * Get CuckooHashTable statistics for monitoring.
     */
    public Map<String, Object> getCuckooStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("prefixIndex", prefixIndex.stats());
        stats.put("hashToIntentType", hashToIntentType.stats());
        return stats;
    }

    /**
     * Match result record.
     */
    public record MatchResult(SemanticHash hash, String intentType, double confidence) {}

    /**
     * Internal hash entry for the index.
     */
    private record HashEntry(SemanticHash hash, String intentType, String exampleQuery) {}
}
