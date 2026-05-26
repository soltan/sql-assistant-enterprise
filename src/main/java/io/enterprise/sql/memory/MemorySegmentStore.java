package io.enterprise.sql.memory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Off-heap vector storage using Panama Memory API + MemoryLayout.
 * Stores float vectors in a contiguous memory segment for cache-friendly access.
 * Zero GC pressure — all data lives off-heap.
 *
 * Memory layout per vector entry:
 *   [id: long(8 bytes)] [padding: 4 bytes] [dim: int(4 bytes)] [components: float[dim]]
 *
 * Total entry size = 16 + dim * 4 bytes
 */
public class MemorySegmentStore implements AutoCloseable {

    private static final int ID_OFFSET = 0;
    private static final int DIM_OFFSET = 8;
    private static final int VECTOR_OFFSET = 16;

    private final Arena arena;
    private final int dimension;
    private final int entrySize;
    private final long maxEntries;
    private final MemorySegment segment;
    private final AtomicLong entryCount;

    // VarHandle for structured access
    private final VarHandle idHandle;
    private final VarHandle dimHandle;
    private final VarHandle componentHandle;

    /**
     * Create a new off-heap vector store.
     *
     * @param dimension  vector dimensionality (e.g., 128, 256, 512)
     * @param maxEntries maximum number of vectors to store
     */
    public MemorySegmentStore(int dimension, long maxEntries) {
        this.dimension = dimension;
        this.maxEntries = maxEntries;
        this.entrySize = VECTOR_OFFSET + dimension * Float.BYTES;
        this.entryCount = new AtomicLong(0);

        // Define structured memory layout
        MemoryLayout entryLayout = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("id"),
            MemoryLayout.paddingLayout(4),
            ValueLayout.JAVA_INT.withName("dim"),
            MemoryLayout.sequenceLayout(dimension, ValueLayout.JAVA_FLOAT).withName("components")
        );

        this.idHandle = entryLayout.varHandle(MemoryLayout.PathElement.groupElement("id"));
        this.dimHandle = entryLayout.varHandle(MemoryLayout.PathElement.groupElement("dim"));
        this.componentHandle = entryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("components"),
            MemoryLayout.PathElement.sequenceElement()
        );

        long totalSize = (long) entrySize * maxEntries;
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(totalSize);

        // Zero-initialize the segment
        segment.fill((byte) 0);
    }

    /**
     * Store a vector with the given ID. Returns the entry index.
     */
    public long store(long id, float[] vector) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                "Vector dimension mismatch: expected %d, got %d".formatted(dimension, vector.length));
        }
        long index = entryCount.getAndIncrement();
        if (index >= maxEntries) {
            throw new IllegalStateException("Vector store is full (max %d entries)".formatted(maxEntries));
        }

        long offset = index * entrySize;
        MemorySegment entry = segment.asSlice(offset, entrySize);

        idHandle.set(entry, 0L, id);
        dimHandle.set(entry, 0L, dimension);
        for (int i = 0; i < dimension; i++) {
            componentHandle.set(entry, 0L, (long) i, vector[i]);
        }

        return index;
    }

    /**
     * Read a vector by entry index.
     */
    public float[] readVector(long index) {
        if (index < 0 || index >= entryCount.get()) {
            throw new IndexOutOfBoundsException("Index: %d, Size: %d".formatted(index, entryCount.get()));
        }

        long offset = index * entrySize;
        MemorySegment entry = segment.asSlice(offset, entrySize);

        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (float) componentHandle.get(entry, 0L, (long) i);
        }
        return vector;
    }

    /**
     * Read the ID of an entry by index.
     */
    public long readId(long index) {
        if (index < 0 || index >= entryCount.get()) {
            throw new IndexOutOfBoundsException("Index: %d, Size: %d".formatted(index, entryCount.get()));
        }
        long offset = index * entrySize;
        MemorySegment entry = segment.asSlice(offset, entrySize);
        return (long) idHandle.get(entry, 0L);
    }

    /**
     * Compute dot product between a query vector and a stored vector.
     * Direct memory access — no intermediate array allocation.
     */
    public float dotProduct(long index, float[] query) {
        if (query.length != dimension) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        long offset = index * entrySize + VECTOR_OFFSET;
        float sum = 0.0f;
        for (int i = 0; i < dimension; i++) {
            float component = segment.get(ValueLayout.JAVA_FLOAT, offset + (long) i * Float.BYTES);
            sum += component * query[i];
        }
        return sum;
    }

    /**
     * Compute cosine similarity between a query vector and a stored vector.
     */
    public float cosineSimilarity(long index, float[] query) {
        long offset = index * entrySize + VECTOR_OFFSET;
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;

        for (int i = 0; i < dimension; i++) {
            float component = segment.get(ValueLayout.JAVA_FLOAT, offset + (long) i * Float.BYTES);
            dotProduct += component * query[i];
            normA += component * component;
            normB += query[i] * query[i];
        }

        if (normA == 0.0f || normB == 0.0f) return 0.0f;
        return dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Bulk scan: find the top-K nearest vectors by cosine similarity.
     * Brute-force but cache-friendly due to contiguous memory layout.
     */
    public long[] findTopK(float[] query, int k) {
        long count = entryCount.get();
        record Score(long index, float similarity) {}
        var heap = new java.util.PriorityQueue<Score>(
            java.util.Comparator.comparingDouble(Score::similarity));

        for (long i = 0; i < count; i++) {
            float sim = cosineSimilarity(i, query);
            if (heap.size() < k) {
                heap.offer(new Score(i, sim));
            } else if (sim > heap.peek().similarity) {
                heap.poll();
                heap.offer(new Score(i, sim));
            }
        }

        return heap.stream()
            .sorted(java.util.Comparator.comparingDouble(Score::similarity).reversed())
            .mapToLong(Score::index)
            .toArray();
    }

    /**
     * Memory usage in bytes.
     */
    public long memoryUsage() {
        return (long) entrySize * entryCount.get();
    }

    /**
     * Current number of stored vectors.
     */
    public long size() {
        return entryCount.get();
    }

    /**
     * Vector dimension.
     */
    public int dimension() {
        return dimension;
    }

    @Override
    public void close() {
        arena.close();
    }
}
