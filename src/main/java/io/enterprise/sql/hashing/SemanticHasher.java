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
 * Also maintains a hash cluster index for O(1) approximate lookup.
 */
public class SemanticHasher {

    /** Cluster radius in Hamming distance for grouping similar hashes */
    private static final int CLUSTER_RADIUS = 8;

    /** Inverted index: hash prefix → cluster of known hashes */
    private final ConcurrentHashMap<Long, List<HashEntry>> prefixIndex;

    /** Known intent hashes for lookup */
    private final ConcurrentHashMap<SemanticHash, String> hashToIntentType;

    public SemanticHasher() {
        this.prefixIndex = new ConcurrentHashMap<>();
        this.hashToIntentType = new ConcurrentHashMap<>();
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
        long prefix = hash.high() >>> 48;
        prefixIndex.computeIfAbsent(prefix, k -> new CopyOnWriteArrayList<>())
            .add(new HashEntry(hash, intentType, exampleQuery));
    }

    /**
     * Hash a query and find the best matching intent type.
     *
     * @param query user query
     * @return match result with confidence
     */
    public MatchResult match(String query) {
        SemanticHash queryHash = SimHash.computeFromQuery(query);

        // Phase 1: Prefix filtering — only check clusters with same 16-bit prefix
        long prefix = queryHash.high() >>> 48;
        List<HashEntry> candidates = prefixIndex.getOrDefault(prefix, List.of());

        // Also check neighboring prefixes (±1 for tolerance)
        candidates = new ArrayList<>(candidates);
        candidates.addAll(prefixIndex.getOrDefault(prefix - 1, List.of()));
        candidates.addAll(prefixIndex.getOrDefault(prefix + 1, List.of()));

        if (candidates.isEmpty()) {
            return new MatchResult(queryHash, "Unknown", 0.0);
        }

        // Phase 2: Hamming distance ranking
        MatchResult best = null;
        for (HashEntry entry : candidates) {
            int distance = queryHash.hammingDistance(entry.hash);
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
        return hashToIntentType.size();
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
