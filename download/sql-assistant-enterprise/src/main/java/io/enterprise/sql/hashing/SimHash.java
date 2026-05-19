package io.enterprise.sql.hashing;

import io.enterprise.sql.model.SemanticHash;
import java.util.*;

/**
 * SimHash implementation for semantic hashing of SQL queries.
 * Locality-sensitive: similar queries produce hashes with small Hamming distance.
 * Deterministic: same input always produces same hash.
 * O(d) time complexity where d = number of features.
 *
 * Algorithm:
 * 1. Extract features (tokens, n-grams, keyword patterns) from query
 * 2. Hash each feature with multiple hash functions
 * 3. Accumulate bit votes across all features
 * 4. Threshold to produce final 128-bit hash
 */
public final class SimHash {

    /** Number of bits in the hash = 128 */
    private static final int HASH_BITS = 128;

    /** Number of hash functions for feature hashing */
    private static final int NUM_HASH_FUNCTIONS = 8;

    /** Pre-computed random seeds for hash functions */
    private static final long[] HASH_SEEDS = {
        0x9e3779b97f4a7c15L, 0x94d049bb133111ebL,
        0x3c79ac57e3b96f62L, 0x5bd1e9959e8c94c6L,
        0xc13b5e4e1e3b2683L, 0x6c62272e07b9a543L,
        0x2545f0d1f7a013a4L, 0x1a7ebbf2dbe5a7b8L
    };

    /** SQL keyword weight table — keywords get higher influence on hash */
    private static final Map<String, Double> KEYWORD_WEIGHTS = Map.ofEntries(
        Map.entry("select", 3.0), Map.entry("from", 2.5),
        Map.entry("where", 2.5),  Map.entry("insert", 3.0),
        Map.entry("update", 3.0), Map.entry("delete", 3.0),
        Map.entry("create", 3.0), Map.entry("alter", 3.0),
        Map.entry("drop", 3.0),   Map.entry("join", 2.8),
        Map.entry("inner", 2.0),  Map.entry("left", 2.0),
        Map.entry("right", 2.0),  Map.entry("outer", 2.0),
        Map.entry("group", 2.5),  Map.entry("order", 2.5),
        Map.entry("having", 2.5), Map.entry("count", 2.5),
        Map.entry("sum", 2.5),    Map.entry("avg", 2.5),
        Map.entry("min", 2.5),    Map.entry("max", 2.5),
        Map.entry("and", 1.5),    Map.entry("or", 1.5),
        Map.entry("not", 1.5),    Map.entry("in", 1.5),
        Map.entry("like", 1.5),   Map.entry("between", 1.5),
        Map.entry("explain", 2.8), Map.entry("schema", 2.5),
        Map.entry("table", 2.0),  Map.entry("column", 2.0),
        Map.entry("index", 2.0),  Map.entry("constraint", 2.0),
        Map.entry("all", 1.5),    Map.entry("distinct", 2.0),
        Map.entry("limit", 1.8),  Map.entry("offset", 1.5),
        Map.entry("union", 2.5),  Map.entry("intersect", 2.5),
        Map.entry("except", 2.5), Map.entry("as", 1.2),
        Map.entry("on", 1.5),     Map.entry("into", 2.0),
        Map.entry("values", 2.0), Map.entry("set", 2.0)
    );

    private SimHash() {} // utility class

    /**
     * Compute a 128-bit SimHash from a list of weighted features.
     *
     * @param features map of feature string → weight
     * @return semantic hash
     */
    public static SemanticHash compute(Map<String, Double> features) {
        int[] bits = new int[HASH_BITS];

        for (var entry : features.entrySet()) {
            String feature = entry.getKey();
            double weight = entry.getValue();

            // Hash the feature with each hash function to get bit positions
            long[] hashes = hashFeature(feature);

            // Accumulate weighted votes
            for (int i = 0; i < HASH_BITS; i++) {
                int hashIndex = i / Long.SIZE;
                int bitIndex = i % Long.SIZE;
                boolean bitSet = ((hashes[hashIndex] >>> bitIndex) & 1L) == 1L;
                bits[i] += bitSet ? weight : -weight;
            }
        }

        // Threshold to produce final hash
        long high = 0L;
        long low = 0L;

        for (int i = 0; i < Long.SIZE; i++) {
            if (bits[i] > 0) high |= (1L << i);
        }
        for (int i = 0; i < Long.SIZE; i++) {
            if (bits[Long.SIZE + i] > 0) low |= (1L << i);
        }

        return new SemanticHash(high, low);
    }

    /**
     * Compute SimHash from a raw query string.
     * Extracts features automatically: tokens + bigrams + keyword patterns.
     */
    public static SemanticHash computeFromQuery(String query) {
        if (query == null || query.isBlank()) return SemanticHash.ZERO;

        String normalized = normalize(query);
        Map<String, Double> features = new LinkedHashMap<>();

        // 1. Token features (unigrams)
        String[] tokens = normalized.split("\\s+");
        for (String token : tokens) {
            if (!token.isEmpty()) {
                double weight = KEYWORD_WEIGHTS.getOrDefault(token, 1.0);
                features.merge(token, weight, Double::sum);
            }
        }

        // 2. Bigram features (captures phrase structure)
        for (int i = 0; i < tokens.length - 1; i++) {
            String bigram = tokens[i] + "_" + tokens[i + 1];
            features.merge(bigram, 1.5, Double::sum);
        }

        // 3. Trigram features (captures intent patterns like "select * from")
        for (int i = 0; i < tokens.length - 2; i++) {
            String trigram = tokens[i] + "_" + tokens[i + 1] + "_" + tokens[i + 2];
            features.merge(trigram, 2.0, Double::sum);
        }

        // 4. Structural pattern features
        addStructuralFeatures(normalized, features);

        return compute(features);
    }

    /**
     * Add structural pattern features (wildcard, parentheses, operators, etc.).
     */
    private static void addStructuralFeatures(String normalized, Map<String, Double> features) {
        // Wildcard pattern: "select *"
        if (normalized.contains("select *") || normalized.contains("select  *")) {
            features.merge("PATTERN:wildcard_select", 3.0, Double::sum);
        }

        // Subquery pattern: nested select
        if (normalized.contains("select") && normalized.indexOf("select") !=
            normalized.lastIndexOf("select")) {
            features.merge("PATTERN:subquery", 4.0, Double::sum);
        }

        // Aggregation pattern
        if (normalized.contains("count(") || normalized.contains("sum(") ||
            normalized.contains("avg(") || normalized.contains("min(") ||
            normalized.contains("max(")) {
            features.merge("PATTERN:aggregate", 3.5, Double::sum);
        }

        // Join pattern
        if (normalized.contains("join")) {
            features.merge("PATTERN:join", 3.5, Double::sum);
        }

        // Filter pattern
        if (normalized.contains("where")) {
            features.merge("PATTERN:filter", 2.5, Double::sum);
        }

        // Group by pattern
        if (normalized.contains("group by")) {
            features.merge("PATTERN:grouping", 3.0, Double::sum);
        }

        // Limit pattern
        if (normalized.contains("limit")) {
            features.merge("PATTERN:limit", 2.0, Double::sum);
        }

        // Order pattern
        if (normalized.contains("order by")) {
            features.merge("PATTERN:ordering", 2.0, Double::sum);
        }
    }

    /**
     * Hash a feature string using multiple hash functions.
     * Returns two longs (128 bits) of hash data.
     */
    private static long[] hashFeature(String feature) {
        long[] hashes = new long[2];
        long h1 = murmurHash3(feature, HASH_SEEDS[0]);
        long h2 = murmurHash3(feature, HASH_SEEDS[1]);
        hashes[0] = h1;
        hashes[1] = h2;
        return hashes;
    }

    /**
     * Simplified MurmurHash3 for 64-bit hash values.
     */
    private static long murmurHash3(String data, long seed) {
        long h = seed;
        for (int i = 0; i < data.length(); i++) {
            h ^= data.charAt(i);
            h *= 0x5bd1e9959e8c94c6L;
            h ^= h >>> 47;
        }
        h *= 0x5bd1e9959e8c94c6L;
        h ^= h >>> 47;
        return h;
    }

    /**
     * Normalize a query: lowercase, collapse whitespace, remove punctuation except *.
     */
    private static String normalize(String query) {
        return query.toLowerCase(Locale.ROOT)
            .replaceAll("[^a-z0-9_*\\s]", " ")
            .replaceAll("\\s+", " ")
            .trim();
    }

    /**
     * Compute the Hamming distance between two semantic hashes.
     * Convenience delegate.
     */
    public static int hammingDistance(SemanticHash a, SemanticHash b) {
        return a.hammingDistance(b);
    }

    /**
     * Find all hashes within maxDistance from a target in a collection.
     * Linear scan but O(1) per comparison.
     */
    public static List<SemanticHash> findNearby(SemanticHash target,
                                                 Collection<SemanticHash> hashes,
                                                 int maxDistance) {
        return hashes.stream()
            .filter(h -> h.hammingDistance(target) <= maxDistance)
            .toList();
    }
}
