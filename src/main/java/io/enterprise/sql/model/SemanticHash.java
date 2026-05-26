package io.enterprise.sql.model;

/**
 * Value-like semantic hash representing the intent signature of a query.
 * Two queries with similar intent produce hashes with small Hamming distance.
 *
 * Uses a 128-bit hash stored as two longs for zero-allocation comparison.
 * Designed for Valhalla value type migration (identity-free, flat layout).
 *
 * @param high upper 64 bits of the hash
 * @param low  lower 64 bits of the hash
 */
public record SemanticHash(long high, long low) implements Comparable<SemanticHash> {

    /**
     * Computes Hamming distance between two semantic hashes.
     * Lower distance = higher semantic similarity.
     * Time complexity: O(1) — uses Long.bitCount for popcount.
     */
    public int hammingDistance(SemanticHash other) {
        long xorHigh = this.high ^ other.high;
        long xorLow  = this.low  ^ other.low;
        return Long.bitCount(xorHigh) + Long.bitCount(xorLow);
    }

    /**
     * Returns similarity as a value in [0.0, 1.0].
     * 1.0 = identical, 0.0 = completely different.
     */
    public double similarity(SemanticHash other) {
        return 1.0 - (hammingDistance(other) / 128.0);
    }

    /**
     * Checks if this hash is within a Hamming distance threshold.
     * Used for fast approximate matching in the intent graph.
     */
    public boolean isWithin(SemanticHash other, int maxDistance) {
        return hammingDistance(other) <= maxDistance;
    }

    @Override
    public int compareTo(SemanticHash other) {
        int cmp = Long.compareUnsigned(this.high, other.high);
        return cmp != 0 ? cmp : Long.compareUnsigned(this.low, other.low);
    }

    /**
     * Combine with another hash via XOR for clustering.
     */
    public SemanticHash xor(SemanticHash other) {
        return new SemanticHash(this.high ^ other.high, this.low ^ other.low);
    }

    /**
     * Zero hash sentinel value.
     */
    public static SemanticHash ZERO = new SemanticHash(0L, 0L);
}
