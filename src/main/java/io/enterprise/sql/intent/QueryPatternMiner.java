package io.enterprise.sql.intent;

import io.enterprise.sql.model.SemanticHash;
import io.enterprise.sql.hashing.SimHash;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Statistical query pattern miner for continuous improvement.
 * Discovers common query patterns, tracks their frequency, and builds
 * a frequency-weighted intent model that prioritizes common patterns.
 *
 * No LLM, no external dependency — pure statistics.
 *
 * Mining strategies:
 * 1. Frequency tracking: count how often each intent type appears
 * 2. Pattern clustering: group similar queries by semantic hash
 * 3. Bigram mining: discover common query bigrams and their associated intents
 * 4. Error tracking: identify queries that were resolved with low confidence
 * 5. Temporal patterns: track query patterns over time windows
 *
 * The mined patterns feed back into:
 * - IntentResolver: frequency-weighted scoring (common intents get a boost)
 * - EnsembleResolver: strategy weight adjustment based on historical accuracy
 * - AdaptiveConfidence: calibration curve updates
 *
 * All state in ConcurrentHashMap — thread-safe.
 */
public class QueryPatternMiner {

    /** Intent frequency counts */
    private final ConcurrentHashMap<String, AtomicLong> intentFrequency;

    /** Hash-based pattern clusters */
    private final ConcurrentHashMap<SemanticHash, PatternCluster> hashClusters;

    /** Bigram → intent association counts */
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, AtomicLong>> bigramIntentCounts;

    /** Low-confidence queries for analysis */
    private final ConcurrentHashMap<String, LowConfidenceEntry> lowConfidenceQueries;

    /** Total queries processed */
    private final AtomicLong totalQueries;

    /** Minimum cluster size to be considered a pattern */
    private static final int MIN_CLUSTER_SIZE = 3;

    /** Low confidence threshold */
    private static final double LOW_CONFIDENCE_THRESHOLD = 0.5;

    /**
     * Pattern cluster — groups of similar queries.
     */
    public record PatternCluster(
        SemanticHash representativeHash,
        String intentType,
        List<String> exampleQueries,
        long frequency,
        double avgConfidence
    ) {}

    /**
     * Low confidence entry — queries that need attention.
     */
    public record LowConfidenceEntry(
        String query,
        String resolvedIntent,
        double confidence,
        long timestamp
    ) {}

    /**
     * Mined pattern result.
     */
    public record MinedPattern(
        String patternKey,
        String intentType,
        long frequency,
        double avgConfidence,
        double frequencyWeight
    ) {}

    public QueryPatternMiner() {
        this.intentFrequency = new ConcurrentHashMap<>();
        this.hashClusters = new ConcurrentHashMap<>();
        this.bigramIntentCounts = new ConcurrentHashMap<>();
        this.lowConfidenceQueries = new ConcurrentHashMap<>();
        this.totalQueries = new AtomicLong(0);
    }

    /**
     * Record a query resolution event.
     * This is the main input: every resolved query updates the pattern model.
     */
    public void record(String query, String intentType, double confidence) {
        totalQueries.incrementAndGet();

        // 1. Update intent frequency
        intentFrequency.computeIfAbsent(intentType, k -> new AtomicLong(0)).incrementAndGet();

        // 2. Update hash cluster
        SemanticHash hash = SimHash.computeFromQuery(query);
        hashClusters.compute(hash, (h, existing) -> {
            if (existing == null) {
                return new PatternCluster(h, intentType, List.of(query), 1, confidence);
            }
            // Merge: update frequency and average confidence
            long newFreq = existing.frequency() + 1;
            double newAvgConf = (existing.avgConfidence() * existing.frequency() + confidence) / newFreq;
            List<String> newExamples = existing.exampleQueries().size() < 5
                ? new ArrayList<>(existing.exampleQueries()) {{ add(query); }}
                : existing.exampleQueries();
            return new PatternCluster(h, intentType, newExamples, newFreq, newAvgConf);
        });

        // 3. Update bigram-intent associations
        String[] tokens = query.toLowerCase(Locale.ROOT).split("\\s+");
        for (int i = 0; i < tokens.length - 1; i++) {
            String bigram = tokens[i] + "_" + tokens[i + 1];
            bigramIntentCounts
                .computeIfAbsent(bigram, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(intentType, k -> new AtomicLong(0))
                .incrementAndGet();
        }

        // 4. Track low-confidence queries
        if (confidence < LOW_CONFIDENCE_THRESHOLD) {
            lowConfidenceQueries.put(query, new LowConfidenceEntry(
                query, intentType, confidence, System.currentTimeMillis()));
        }
    }

    /**
     * Get frequency-weighted boost for an intent type.
     * Common intents get a boost proportional to their log-frequency.
     * This ensures that common intents (like SELECT) are preferred over
     * rare ones (like DROP TABLE) when other signals are ambiguous.
     */
    public double frequencyBoost(String intentType) {
        long freq = intentFrequency.getOrDefault(intentType, new AtomicLong(0)).get();
        if (freq == 0) return 1.0;
        // Log-frequency boost: 1.0 + log(freq) / 10
        // At freq=1: boost = 1.0, at freq=100: boost ≈ 1.46, at freq=10000: boost ≈ 1.92
        return 1.0 + Math.log(freq) / 10.0;
    }

    /**
     * Predict the intent type from a query's bigrams.
     * Uses historical bigram-intent associations as a naive Bayes predictor.
     */
    public Map<String, Double> bigramPredict(String query) {
        String[] tokens = query.toLowerCase(Locale.ROOT).split("\\s+");
        Map<String, Double> scores = new LinkedHashMap<>();

        for (int i = 0; i < tokens.length - 1; i++) {
            String bigram = tokens[i] + "_" + tokens[i + 1];
            ConcurrentHashMap<String, AtomicLong> intentCounts = bigramIntentCounts.get(bigram);
            if (intentCounts != null) {
                long totalForBigram = intentCounts.values().stream()
                    .mapToLong(AtomicLong::get).sum();
                for (var entry : intentCounts.entrySet()) {
                    double prob = (double) entry.getValue().get() / totalForBigram;
                    scores.merge(entry.getKey(), prob, Double::sum);
                }
            }
        }

        // Normalize
        double total = scores.values().stream().mapToDouble(Double::doubleValue).sum();
        if (total > 0) {
            scores.replaceAll((k, v) -> v / total);
        }

        return scores;
    }

    /**
     * Get all mined patterns with sufficient frequency.
     */
    public List<MinedPattern> getPatterns() {
        long total = totalQueries.get();
        if (total == 0) return List.of();

        return hashClusters.values().stream()
            .filter(c -> c.frequency() >= MIN_CLUSTER_SIZE)
            .map(c -> new MinedPattern(
                c.representativeHash().toString(),
                c.intentType(),
                c.frequency(),
                c.avgConfidence(),
                (double) c.frequency() / total
            ))
            .sorted(Comparator.comparingLong(MinedPattern::frequency).reversed())
            .collect(Collectors.toList());
    }

    /**
     * Get the most common intent types ranked by frequency.
     */
    public List<Map.Entry<String, Long>> getIntentRanking() {
        return intentFrequency.entrySet().stream()
            .map(e -> Map.entry(e.getKey(), e.getValue().get()))
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .collect(Collectors.toList());
    }

    /**
     * Get low-confidence queries for analysis.
     */
    public Collection<LowConfidenceEntry> getLowConfidenceQueries() {
        return lowConfidenceQueries.values();
    }

    /**
     * Get mining statistics.
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalQueries", totalQueries.get());
        stats.put("uniqueIntents", intentFrequency.size());
        stats.put("uniqueHashClusters", hashClusters.size());
        stats.put("uniqueBigrams", bigramIntentCounts.size());
        stats.put("lowConfidenceQueries", lowConfidenceQueries.size());
        stats.put("significantPatterns",
            hashClusters.values().stream()
                .filter(c -> c.frequency() >= MIN_CLUSTER_SIZE)
                .count());
        return stats;
    }

    /**
     * Get the total number of queries processed.
     */
    public long totalQueries() {
        return totalQueries.get();
    }
}
