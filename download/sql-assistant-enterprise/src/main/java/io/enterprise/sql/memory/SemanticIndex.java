package io.enterprise.sql.memory;

import io.enterprise.sql.model.SemanticHash;
import java.lang.foreign.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Semantic index that maps SemanticHash → stored entries.
 * Uses Panama off-heap storage for the vector representations
 * and a concurrent hash map for the hash → index lookup.
 *
 * Supports fast approximate lookup via Hamming distance filtering.
 */
public class SemanticIndex implements AutoCloseable {

    private final MemorySegmentStore vectorStore;
    private final ConcurrentHashMap<SemanticHash, List<Long>> hashToIndices;
    private final ConcurrentHashMap<Long, SemanticHash> idToHash;

    /**
     * Create a semantic index with the given vector dimension and capacity.
     */
    public SemanticIndex(int dimension, long capacity) {
        this.vectorStore = new MemorySegmentStore(dimension, capacity);
        this.hashToIndices = new ConcurrentHashMap<>();
        this.idToHash = new ConcurrentHashMap<>();
    }

    /**
     * Index a vector with its semantic hash and unique ID.
     */
    public void index(long id, SemanticHash hash, float[] vector) {
        long index = vectorStore.store(id, vector);
        hashToIndices.computeIfAbsent(hash, k -> new CopyOnWriteArrayList<>()).add(index);
        idToHash.put(id, hash);
    }

    /**
     * Find entries whose semantic hash is within the given Hamming distance.
     * First filters by Hamming distance on hashes, then re-ranks by cosine similarity.
     */
    public List<SearchResult> search(SemanticHash queryHash, float[] queryVector,
                                      int maxDistance, int topK) {
        // Phase 1: Hash-level filtering
        List<Long> candidates = new ArrayList<>();
        for (var entry : hashToIndices.entrySet()) {
            if (entry.getKey().isWithin(queryHash, maxDistance)) {
                candidates.addAll(entry.getValue());
            }
        }

        // Phase 2: Re-rank by cosine similarity
        record Score(long index, float similarity) {}
        var scored = candidates.stream()
            .map(idx -> new Score(idx, vectorStore.cosineSimilarity(idx, queryVector)))
            .sorted(Comparator.comparingDouble(Score::similarity).reversed())
            .limit(topK)
            .toList();

        return scored.stream()
            .map(s -> new SearchResult(
                vectorStore.readId(s.index),
                s.similarity,
                vectorStore.readVector(s.index)
            ))
            .toList();
    }

    /**
     * Get the semantic hash for a given ID.
     */
    public Optional<SemanticHash> getHash(long id) {
        return Optional.ofNullable(idToHash.get(id));
    }

    /**
     * Number of indexed entries.
     */
    public long size() {
        return vectorStore.size();
    }

    /**
     * Close the underlying vector store.
     */
    @Override
    public void close() {
        vectorStore.close();
    }

    /**
     * Search result record.
     */
    public record SearchResult(long id, float similarity, float[] vector) {}
}
