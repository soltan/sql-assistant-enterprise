package io.enterprise.sql.nlp;

import io.enterprise.sql.vector.VectorOps;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Enhanced Embedding Engine V2 — dramatically improved semantic representation.
 * NO LLM, NO neural network, NO external dependency.
 *
 * Improvements over V1:
 * 1. Co-occurrence encoding: captures word relationships from a built-in corpus
 * 2. Syntactic structure encoding: SVO parse tree shape influences embedding dimensions
 * 3. Thesaurus-expanded features: synonym-aware feature generation
 * 4. Position-sensitive hashing: word order matters more
 * 5. Multi-resolution hashing: multiple hash functions at different granularities
 * 6. Semantic role encoding: subject/object/modifier roles get different offsets
 *
 * These improvements make the embeddings much more discriminative:
 * - "select name from users" and "show me the names of all users" → very similar
 * - "select name from users" and "delete name from users" → very different
 * - "select name from users" and "select name from orders" → moderately different
 *
 * Deterministic: same input always produces same embedding.
 */
public class EnhancedEmbeddingEngine {

    private final int dimension;
    private final EmbeddingEngine baseEngine;
    private final ConcurrentHashMap<String, float[]> embeddingCache;

    /** Co-occurrence matrix: word → {context_word → strength} */
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, Double>> cooccurrenceMatrix;

    /** Random projection matrix (deterministic) */
    private final float[][] projectionMatrix;

    /** Position weight decay factor */
    private static final double POSITION_DECAY = 0.95;

    /** Co-occurrence contribution weight */
    private static final double COOCCURRENCE_WEIGHT = 0.4;

    /** Syntactic structure contribution weight */
    private static final double SYNTACTIC_WEIGHT = 0.3;

    /** Thesaurus expansion contribution weight */
    private static final double THESAURUS_WEIGHT = 0.2;

    /** Base token contribution weight */
    private static final double BASE_WEIGHT = 0.5;

    public EnhancedEmbeddingEngine(int dimension) {
        this.dimension = dimension;
        this.baseEngine = new EmbeddingEngine(dimension);
        this.embeddingCache = new ConcurrentHashMap<>();
        this.cooccurrenceMatrix = new ConcurrentHashMap<>();
        this.projectionMatrix = generateProjectionMatrix(dimension);
        initializeCooccurrenceMatrix();
    }

    /**
     * Initialize the co-occurrence matrix with built-in SQL domain knowledge.
     * These represent statistical relationships between SQL terms that would
     * normally be learned from a large corpus, but are pre-built here
     * since we have no training data and no LLM.
     */
    private void initializeCooccurrenceMatrix() {
        // Define co-occurrence groups: words that frequently appear together
        String[][] cooccurrenceGroups = {
            {"select", "from", "where", "columns", "table"},
            {"insert", "into", "values", "row", "add"},
            {"update", "set", "where", "modify", "change"},
            {"delete", "from", "where", "remove", "row"},
            {"create", "table", "columns", "types", "new"},
            {"join", "on", "left", "right", "inner", "outer"},
            {"group", "by", "aggregate", "count", "sum"},
            {"order", "by", "sort", "asc", "desc"},
            {"count", "sum", "avg", "min", "max", "aggregate"},
            {"where", "filter", "condition", "and", "or"},
            {"limit", "top", "first", "rows"},
            {"distinct", "unique", "different"},
            {"index", "performance", "fast", "query"},
            {"null", "empty", "missing", "value"},
            {"primary", "key", "unique", "identifier"},
            {"foreign", "key", "reference", "relation"},
            {"explain", "plan", "execution", "analyze"},
            {"schema", "structure", "columns", "describe"}
        };

        for (String[] group : cooccurrenceGroups) {
            for (int i = 0; i < group.length; i++) {
                for (int j = 0; j < group.length; j++) {
                    if (i != j) {
                        cooccurrenceMatrix
                            .computeIfAbsent(group[i], k -> new ConcurrentHashMap<>())
                            .put(group[j], 0.8); // strong co-occurrence
                    }
                }
            }
        }
    }

    /**
     * Compute an enhanced embedding for the given text.
     * Combines base embedding with co-occurrence, syntactic, and thesaurus features.
     */
    public float[] embed(String text) {
        if (text == null || text.isBlank()) {
            return new float[dimension];
        }

        String normalized = text.toLowerCase(Locale.ROOT).trim();
        return embeddingCache.computeIfAbsent(normalized, this::computeEnhancedEmbedding);
    }

    /**
     * Compute the enhanced embedding.
     */
    private float[] computeEnhancedEmbedding(String text) {
        float[] embedding = new float[dimension];

        // Phase 1: Base token-level embedding (from V1)
        float[] baseEmb = baseEngine.embed(text);
        for (int d = 0; d < dimension; d++) {
            embedding[d] += baseEmb[d] * BASE_WEIGHT;
        }

        // Phase 2: Co-occurrence enrichment
        addCooccurrenceFeatures(text, embedding);

        // Phase 3: Syntactic structure encoding
        addSyntacticFeatures(text, embedding);

        // Phase 4: Thesaurus-expanded features
        addThesaurusFeatures(text, embedding);

        // Phase 5: Position-sensitive encoding
        addPositionFeatures(text, embedding);

        // Phase 6: Random projection for mixing
        float[] projected = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            float sum = 0.0f;
            for (int s = 0; s < dimension; s++) {
                sum += projectionMatrix[d][s] * embedding[s];
            }
            projected[d] = sum;
        }

        // Phase 7: Normalize to unit length
        VectorOps.normalize(projected);

        return projected;
    }

    /**
     * Add co-occurrence-based features.
     * For each token, add contributions from its co-occurring words.
     * This creates a "spreading activation" effect: if "select" appears,
     * the dimensions associated with "from", "where", etc. also get activated.
     */
    private void addCooccurrenceFeatures(String text, float[] embedding) {
        String[] tokens = text.split("\\s+");
        for (String token : tokens) {
            if (token.isEmpty()) continue;
            ConcurrentHashMap<String, Double> coocs = cooccurrenceMatrix.get(token);
            if (coocs != null) {
                for (var entry : coocs.entrySet()) {
                    String coWord = entry.getKey();
                    double strength = entry.getValue();
                    long hash = deterministicHash(coWord);
                    for (int d = 0; d < dimension; d++) {
                        long componentSeed = hash ^ (long) d * 0x9e3779b97f4a7c15L;
                        componentSeed = componentSeed * 6364136223846793005L + 1442695040888963407L;
                        float contribution = ((componentSeed >>> 33) % 2000 - 1000) / 1000.0f;
                        embedding[d] += contribution * strength * COOCCURRENCE_WEIGHT;
                    }
                }
            }
        }
    }

    /**
     * Add syntactic structure features.
     * Uses the SyntacticParser to extract SVO structure and encodes
     * the structural roles into dedicated embedding dimensions.
     */
    private void addSyntacticFeatures(String text, float[] embedding) {
        SyntacticParser.ParseResult parse = SyntacticParser.parse(text);
        var structure = parse.structure();

        // Encode verb type into a reserved region
        int verbOffset = dimension / 8;
        int verbCode = structure.verb().ordinal();
        for (int d = 0; d < 8 && verbOffset + d < dimension; d++) {
            embedding[verbOffset + d] += ((verbCode >> d) & 1) * SYNTACTIC_WEIGHT;
        }

        // Encode subject (table) into another reserved region
        if (!structure.subject().isEmpty()) {
            int subjectOffset = dimension / 4;
            long subjectHash = deterministicHash(structure.subject());
            for (int d = 0; d < 16 && subjectOffset + d < dimension; d++) {
                embedding[subjectOffset + d] += ((subjectHash >> d) & 1) * SYNTACTIC_WEIGHT * 0.8;
            }
        }

        // Encode number of conditions (WHERE clause complexity)
        int conditionOffset = dimension / 3;
        int numConditions = structure.conditions().size();
        for (int d = 0; d < 4 && conditionOffset + d < dimension; d++) {
            embedding[conditionOffset + d] += ((numConditions >> d) & 1) * SYNTACTIC_WEIGHT * 0.5;
        }

        // Encode number of modifiers
        int modifierOffset = (int) (dimension * 0.4);
        int numModifiers = structure.modifiers().size();
        for (int d = 0; d < 4 && modifierOffset + d < dimension; d++) {
            embedding[modifierOffset + d] += ((numModifiers >> d) & 1) * SYNTACTIC_WEIGHT * 0.5;
        }

        // Encode number of joins
        int joinOffset = (int) (dimension * 0.5);
        int numJoins = structure.joins().size();
        for (int d = 0; d < 4 && joinOffset + d < dimension; d++) {
            embedding[joinOffset + d] += ((numJoins >> d) & 1) * SYNTACTIC_WEIGHT * 0.6;
        }

        // Structural confidence as a global modulation
        double confMod = structure.structuralConfidence();
        for (int d = 0; d < dimension; d++) {
            embedding[d] *= (1.0 + confMod * 0.2); // boost by up to 20% for well-structured queries
        }
    }

    /**
     * Add thesaurus-expanded features.
     * Uses ThesaurusEngine to expand the query with synonyms,
     * then hashes the expanded features into the embedding.
     */
    private void addThesaurusFeatures(String text, float[] embedding) {
        Map<String, Double> expandedFeatures = ThesaurusEngine.generateExpandedFeatures(text);
        for (var entry : expandedFeatures.entrySet()) {
            String feature = entry.getKey();
            double weight = entry.getValue();
            long hash = deterministicHash(feature);
            int offset = (int) (Math.abs(hash) % (dimension / 2));
            for (int d = 0; d < 4 && offset + d < dimension; d++) {
                float contribution = ((hash >>> (d * 8)) & 0xFF) / 255.0f * 2.0f - 1.0f;
                embedding[offset + d] += contribution * weight * THESAURUS_WEIGHT;
            }
        }
    }

    /**
     * Add position-sensitive features.
     * Earlier tokens get higher weight, capturing the intuition that
     * the first verb ("select") is more important than later modifiers.
     */
    private void addPositionFeatures(String text, float[] embedding) {
        String[] tokens = text.split("\\s+");
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].isEmpty()) continue;
            double posWeight = Math.pow(POSITION_DECAY, i);
            long hash = deterministicHash(tokens[i] + "@" + i);
            int offset = (int) (Math.abs(hash) % (dimension / 4));
            for (int d = 0; d < 4 && offset + d < dimension; d++) {
                float contribution = ((hash >>> (d * 8)) & 0xFF) / 255.0f * 2.0f - 1.0f;
                embedding[offset + d] += contribution * posWeight * 0.3;
            }
        }
    }

    /**
     * Generate a deterministic projection matrix.
     */
    private float[][] generateProjectionMatrix(int dim) {
        float[][] matrix = new float[dim][dim];
        Random rng = new Random(42);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                matrix[i][j] = (rng.nextFloat() - 0.5f) / (float) Math.sqrt(dim);
            }
        }
        return matrix;
    }

    /**
     * Deterministic hash for a string.
     */
    private long deterministicHash(String text) {
        long h = 0x9e3779b97f4a7c15L;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x5bd1e9959e8c94c6L;
            h ^= h >>> 47;
        }
        return h;
    }

    /**
     * Compute similarity between two texts using enhanced embeddings.
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
