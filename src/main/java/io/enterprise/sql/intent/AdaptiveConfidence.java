package io.enterprise.sql.intent;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Adaptive confidence calibration engine.
 *
 * Problem: raw confidence scores from individual strategies are not calibrated.
 * A strategy might report 0.8 confidence but be correct only 60% of the time,
 * while another reports 0.6 but is correct 90% of the time.
 *
 * Solution: maintain per-strategy calibration curves that map raw confidence
 * to actual accuracy. Uses a simple histogram-based approach:
 * - Divide [0, 1] into B bins
 * - For each (strategy, bin) pair, track: total predictions, correct predictions
 * - Calibrated confidence = correct / total for that bin
 * - Apply exponential smoothing for stability with few data points
 *
 * Additionally implements:
 * - Query complexity scoring: complex queries (joins, subqueries) get adjusted thresholds
 * - Domain-specific calibration: different accuracy profiles for SELECT vs INSERT vs etc.
 * - Cold-start prior: for bins with < 5 data points, blend with a prior of 0.5
 *
 * All state is in ConcurrentHashMap — thread-safe, no locks.
 * Zero external dependencies, deterministic calibration after warmup.
 */
public class AdaptiveConfidence {

    /** Number of confidence bins */
    private static final int NUM_BINS = 20;

    /** Minimum samples before using empirical calibration */
    private static final int MIN_SAMPLES = 5;

    /** Smoothing factor for exponential moving average */
    private static final double SMOOTHING = 0.3;

    /** Per-strategy calibration data: strategy → bin → (total, correct) */
    private final ConcurrentHashMap<String, CalibrationBin[]> strategyCalibration;

    /** Per-intent-type calibration data */
    private final ConcurrentHashMap<String, CalibrationBin[]> intentCalibration;

    /** Query complexity weights */
    private static final Map<String, Double> COMPLEXITY_WEIGHTS = Map.of(
        "Select", 1.0,
        "Insert", 0.8,
        "Update", 0.9,
        "Delete", 0.9,
        "Join", 1.3,      // joins are harder → need higher confidence threshold
        "Aggregate", 1.2,  // aggregates have more ambiguity
        "CreateTable", 0.7,
        "Subquery", 1.5    // subqueries are hardest
    );

    /** Calibration bin record */
    static class CalibrationBin {
        volatile long total = 0;
        volatile long correct = 0;
        volatile double smoothed = 0.5; // EMA of accuracy

        synchronized void record(boolean isCorrect) {
            total++;
            correct += isCorrect ? 1 : 0;
            double empirical = (double) correct / total;
            smoothed = SMOOTHING * empirical + (1 - SMOOTHING) * smoothed;
        }

        double calibrated() {
            if (total < MIN_SAMPLES) {
                // Blend with prior
                double prior = 0.5;
                double weight = (double) total / MIN_SAMPLES;
                return weight * smoothed + (1 - weight) * prior;
            }
            return smoothed;
        }
    }

    public AdaptiveConfidence() {
        this.strategyCalibration = new ConcurrentHashMap<>();
        this.intentCalibration = new ConcurrentHashMap<>();
    }

    /**
     * Calibrate a raw confidence score from a specific strategy.
     * Maps raw score through the empirical accuracy curve.
     */
    public double calibrate(String strategy, double rawConfidence) {
        CalibrationBin[] bins = strategyCalibration.computeIfAbsent(
            strategy, k -> createBins());
        int binIndex = toBin(rawConfidence);
        return bins[binIndex].calibrated();
    }

    /**
     * Calibrate for a specific intent type.
     * Some intents are inherently harder and need adjusted thresholds.
     */
    public double calibrateForIntent(String intentType, double rawConfidence) {
        CalibrationBin[] bins = intentCalibration.computeIfAbsent(
            intentType, k -> createBins());
        int binIndex = toBin(rawConfidence);
        return bins[binIndex].calibrated();
    }

    /**
     * Apply complexity adjustment to a confidence score.
     * Complex queries (joins, subqueries) need higher confidence to be accepted.
     */
    public double adjustForComplexity(String intentType, double confidence) {
        double complexity = COMPLEXITY_WEIGHTS.getOrDefault(intentType, 1.0);
        // For complex intents, require higher confidence
        // Adjusted = confidence / complexity (reduces confidence for hard intents)
        return Math.min(1.0, confidence / complexity);
    }

    /**
     * Record an observation: was a prediction correct?
     * Updates both strategy and intent calibration.
     */
    public void record(String strategy, String intentType, double rawConfidence, boolean wasCorrect) {
        // Update strategy calibration
        CalibrationBin[] stratBins = strategyCalibration.computeIfAbsent(
            strategy, k -> createBins());
        stratBins[toBin(rawConfidence)].record(wasCorrect);

        // Update intent calibration
        CalibrationBin[] intentBins = intentCalibration.computeIfAbsent(
            intentType, k -> createBins());
        intentBins[toBin(rawConfidence)].record(wasCorrect);
    }

    /**
     * Determine if a confidence score is above threshold for the given intent.
     * Uses calibrated confidence with complexity adjustment.
     */
    public boolean isConfident(String strategy, String intentType,
                               double rawConfidence, double threshold) {
        double calibrated = calibrate(strategy, rawConfidence);
        double adjusted = adjustForComplexity(intentType, calibrated);
        return adjusted >= threshold;
    }

    /**
     * Get the recommended confidence threshold for a given intent type.
     * More complex intents require higher thresholds.
     */
    public double recommendedThreshold(String intentType) {
        double complexity = COMPLEXITY_WEIGHTS.getOrDefault(intentType, 1.0);
        // Base threshold 0.6, scaled by complexity
        return Math.min(0.95, 0.6 * complexity);
    }

    /**
     * Get calibration statistics.
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();

        strategyCalibration.forEach((strategy, bins) -> {
            long totalSamples = Arrays.stream(bins).mapToLong(b -> b.total).sum();
            long totalCorrect = Arrays.stream(bins).mapToLong(b -> b.correct).sum();
            double overallAccuracy = totalSamples > 0 ? (double) totalCorrect / totalSamples : 0.0;
            stats.put("strategy." + strategy + ".samples", totalSamples);
            stats.put("strategy." + strategy + ".accuracy", String.format("%.3f", overallAccuracy));
        });

        intentCalibration.forEach((intent, bins) -> {
            long totalSamples = Arrays.stream(bins).mapToLong(b -> b.total).sum();
            long totalCorrect = Arrays.stream(bins).mapToLong(b -> b.correct).sum();
            double overallAccuracy = totalSamples > 0 ? (double) totalCorrect / totalSamples : 0.0;
            stats.put("intent." + intent + ".samples", totalSamples);
            stats.put("intent." + intent + ".accuracy", String.format("%.3f", overallAccuracy));
        });

        return stats;
    }

    // ── Helpers ──

    private static CalibrationBin[] createBins() {
        CalibrationBin[] bins = new CalibrationBin[NUM_BINS];
        for (int i = 0; i < NUM_BINS; i++) {
            bins[i] = new CalibrationBin();
        }
        return bins;
    }

    private static int toBin(double confidence) {
        int bin = (int) (confidence * NUM_BINS);
        return Math.max(0, Math.min(NUM_BINS - 1, bin));
    }
}
