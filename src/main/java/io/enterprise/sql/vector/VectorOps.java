package io.enterprise.sql.vector;

import jdk.incubator.vector.*;

/**
 * SIMD-accelerated vector operations using the Java Vector API.
 * All operations are deterministic — no GPU, no external dependency.
 *
 * Uses FloatVector.SPECIES_PREFERRED for best SIMD width on the running CPU
 * (128-bit SSE on x86, 256-bit AVX2, or 512-bit AVX-512 where available).
 */
public final class VectorOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private VectorOps() {} // utility class

    /**
     * Compute dot product using SIMD.
     * Throughput: ~16 floats/cycle on AVX-512, ~8 on AVX2, ~4 on SSE.
     */
    public static float dotProduct(float[] a, float[] b) {
        int length = Math.min(a.length, b.length);
        int i = 0;
        float sum = 0.0f;

        // Vectorized loop
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
        }

        // Tail loop
        for (; i < length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /**
     * Compute cosine similarity using SIMD.
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        int length = Math.min(a.length, b.length);
        int i = 0;
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;

        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            FloatVector products = va.mul(vb);
            dotProduct += products.reduceLanes(VectorOperators.ADD);
            normA += va.mul(va).reduceLanes(VectorOperators.ADD);
            normB += vb.mul(vb).reduceLanes(VectorOperators.ADD);
        }

        for (; i < length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0.0f || normB == 0.0f) return 0.0f;
        return dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Compute Euclidean distance using SIMD.
     */
    public static float euclideanDistance(float[] a, float[] b) {
        int length = Math.min(a.length, b.length);
        int i = 0;
        float sumSquares = 0.0f;

        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            FloatVector diff = va.sub(vb);
            sumSquares += diff.mul(diff).reduceLanes(VectorOperators.ADD);
        }

        for (; i < length; i++) {
            float diff = a[i] - b[i];
            sumSquares += diff * diff;
        }

        return (float) Math.sqrt(sumSquares);
    }

    /**
     * Batch cosine similarity: query vs. multiple vectors.
     * Optimized for repeated similarity searches.
     */
    public static float[] batchCosineSimilarity(float[] query, float[][] vectors) {
        float[] results = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            results[i] = cosineSimilarity(query, vectors[i]);
        }
        return results;
    }

    /**
     * Normalize a vector to unit length in-place using SIMD.
     */
    public static void normalize(float[] vector) {
        int length = vector.length;
        int i = 0;
        float normSq = 0.0f;

        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, vector, i);
            normSq += va.mul(va).reduceLanes(VectorOperators.ADD);
        }
        for (; i < length; i++) {
            normSq += vector[i] * vector[i];
        }

        if (normSq == 0.0f) return;
        float invNorm = 1.0f / (float) Math.sqrt(normSq);

        i = 0;
        FloatVector scaleVec = FloatVector.broadcast(SPECIES, invNorm);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, vector, i);
            va.mul(scaleVec).intoArray(vector, i);
        }
        for (; i < length; i++) {
            vector[i] *= invNorm;
        }
    }

    /**
     * L2 distance squared (avoids sqrt for comparison-only use cases).
     */
    public static float euclideanDistanceSquared(float[] a, float[] b) {
        int length = Math.min(a.length, b.length);
        int i = 0;
        float sumSquares = 0.0f;

        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            FloatVector diff = va.sub(vb);
            sumSquares += diff.mul(diff).reduceLanes(VectorOperators.ADD);
        }

        for (; i < length; i++) {
            float diff = a[i] - b[i];
            sumSquares += diff * diff;
        }

        return sumSquares;
    }

    /**
     * Returns the SIMD vector width being used (number of floats per operation).
     */
    public static int simdWidth() {
        return SPECIES.length();
    }
}
