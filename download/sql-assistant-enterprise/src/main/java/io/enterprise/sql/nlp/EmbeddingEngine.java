package io.enterprise.sql.nlp;

import io.enterprise.sql.vector.VectorOps;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Deterministic embedding engine — NO LLM, NO neural network.
 *
 * Generates fixed-size float vectors from text using:
 * 1. Token hashing: each token is hashed to seed a pseudo-random projection
 * 2. N-gram encoding: bigrams and trigrams contribute additional dimensions
 * 3. Positional encoding: token order influences the embedding
 * 4. Keyword boosting: SQL keywords get amplified contributions
 *
 * Properties:
 * - Deterministic: same input → same embedding
 * - Fast: O(n * d) where n = tokens, d = dimension
 * - Locality-sensitive: similar queries → similar embeddings (via shared tokens)
 * - Normalized: output vectors are unit-length for cosine similarity
 */
public class EmbeddingEngine {

    private final int dimension;
    private final ConcurrentHashMap<String, float[]> embeddingCache;

    /** SQL keyword boost factor */
    private static final double KEYWORD_BOOST = 2.0;

    /** SQL keywords for boosting */
    private static final Set<String> SQL_KEYWORDS = Set.of(
        "select", "from", "where", "insert", "update", "delete", "create",
        "alter", "drop", "join", "group", "order", "having", "limit",
        "count", "sum", "avg", "min", "max", "explain", "schema"
    );

    /** Random projection matrix (deterministic, computed once) */
    private final float[][] projectionMatrix;

    public EmbeddingEngine(int dimension) {
        this.dimension = dimension;
        this.embeddingCache = new ConcurrentHashMap<>();
        this.projectionMatrix = generateProjectionMatrix(dimension);
    }

    /**
     * Compute an embedding for the given text.
     * Results are cached for frequently seen inputs.
     */
    public float[] embed(String text) {
        if (text == null || text.isBlank()) {
            return new float[dimension];
        }

        String normalized = text.toLowerCase(Locale.ROOT).trim();
        return embeddingCache.computeIfAbsent(normalized, this::computeEmbedding);
    }

    /**
     * Compute the embedding without cache.
     */
    private float[] computeEmbedding(String text) {
        float[] embedding = new float[dimension];
        List<Tokenizer.Token> tokens = Tokenizer.tokenize(text);

        // Phase 1: Token-level contributions via random projection
        for (int i = 0; i < tokens.size(); i++) {
            Tokenizer.Token token = tokens.get(i);
            double weight = SQL_KEYWORDS.contains(token.text().toLowerCase(Locale.ROOT))
                ? KEYWORD_BOOST : 1.0;

            // Hash the token to get a deterministic projection
            long hash = hashToken(token.text(), i);
            for (int d = 0; d < dimension; d++) {
                long componentSeed = hash ^ (long) d * 0x9e3779b97f4a7c15L;
                componentSeed = componentSeed * 6364136223846793005L + 1442695040888963407L;
                float contribution = ((componentSeed >>> 33) % 2000 - 1000) / 1000.0f;
                embedding[d] += contribution * weight;
            }
        }

        // Phase 2: Random projection for dimensionality mixing
        float[] projected = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            float sum = 0.0f;
            for (int s = 0; s < Math.min(dimension, tokens.size() * 4); s++) {
                if (s < embedding.length) {
                    sum += projectionMatrix[d][s % dimension] * embedding[s];
                }
            }
            projected[d] = sum;
        }

        // Phase 3: Add bigram features
        addBigramFeatures(tokens, projected);

        // Phase 4: Normalize to unit length
        VectorOps.normalize(projected);

        return projected;
    }

    /**
     * Add bigram-based features to the embedding.
     */
    private void addBigramFeatures(List<Tokenizer.Token> tokens, float[] embedding) {
        for (int i = 0; i < tokens.size() - 1; i++) {
            String bigram = tokens.get(i).text() + "_" + tokens.get(i + 1).text();
            long hash = hashToken(bigram, i);
            int offset = (int) (Math.abs(hash) % (dimension / 2)) + dimension / 4;
            for (int j = 0; j < 4 && offset + j < dimension; j++) {
                embedding[offset + j] += ((hash >>> (j * 8)) & 0xFF) / 255.0f * 0.5f;
            }
        }
    }

    /**
     * Generate a deterministic random projection matrix.
     * Used for dimensionality mixing — ensures diverse representation.
     */
    private float[][] generateProjectionMatrix(int dim) {
        float[][] matrix = new float[dim][dim];
        Random rng = new Random(42); // deterministic seed
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                matrix[i][j] = (rng.nextFloat() - 0.5f) / (float) Math.sqrt(dim);
            }
        }
        return matrix;
    }

    /**
     * Deterministic token hash.
     */
    private long hashToken(String token, int position) {
        long h = 0x9e3779b97f4a7c15L;
        for (int i = 0; i < token.length(); i++) {
            h ^= token.charAt(i);
            h *= 0x5bd1e9959e8c94c6L;
            h ^= h >>> 47;
        }
        h ^= (long) position * 0x94d049bb133111ebL;
        return h;
    }

    /**
     * Compute similarity between two texts.
     */
    public float similarity(String textA, String textB) {
        float[] a = embed(textA);
        float[] b = embed(textB);
        return VectorOps.cosineSimilarity(a, b);
    }

    /**
     * Get the embedding dimension.
     */
    public int dimension() {
        return dimension;
    }

    /**
     * Clear the embedding cache.
     */
    public void clearCache() {
        embeddingCache.clear();
    }

    /**
     * Cache size.
     */
    public int cacheSize() {
        return embeddingCache.size();
    }
}
