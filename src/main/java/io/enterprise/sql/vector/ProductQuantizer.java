package io.enterprise.sql.vector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Product Quantization for memory-efficient vector compression.
 * 
 * Problem: Storing 1M vectors of 128 dimensions = 1M * 128 * 4 bytes = 512 MB.
 * With PQ using 8-bit codes and 32 subspaces: 1M * 32 * 1 byte = 32 MB (16x compression).
 *
 * Algorithm:
 * 1. Split each D-dimensional vector into M subvectors of D/M dimensions
 * 2. For each subspace, run K-means clustering to create K centroids (codebook)
 * 3. Encode each vector as M centroid indices (1 byte each if K=256)
 * 4. Approximate distance = sum of sub-distances from each subspace's codebook
 *
 * This enables:
 * - 16-32x memory compression for vector storage
 * - Fast approximate distance computation via lookup tables
 * - Asymmetric distance computation (ADC) for query-time efficiency
 *
 * No external dependency. Deterministic when seeded. Thread-safe.
 */
public class ProductQuantizer {

    /** Number of subspaces (subvector count) */
    private final int numSubspaces;

    /** Number of centroids per subspace (256 = 1 byte per code) */
    private final int numCentroids;

    /** Dimension of each subvector */
    private final int subDim;

    /** Full vector dimension */
    private final int fullDimension;

    /** Codebooks: [subspace][centroid_index][subDim] */
    private volatile float[][][] codebooks;

    /** Encoded vectors: vectorId → byte[numSubspaces] */
    private final ConcurrentHashMap<Long, byte[]> encodedVectors;

    /** Whether the codebook has been trained */
    private volatile boolean trained;

    /**
     * Create a Product Quantizer.
     *
     * @param fullDimension total vector dimension (e.g., 128)
     * @param numSubspaces  number of subspaces M (e.g., 32, must divide fullDimension evenly)
     * @param numCentroids  centroids per subspace K (e.g., 256 for 1-byte codes)
     */
    public ProductQuantizer(int fullDimension, int numSubspaces, int numCentroids) {
        if (fullDimension % numSubspaces != 0) {
            throw new IllegalArgumentException(
                "fullDimension (%d) must be divisible by numSubspaces (%d)"
                    .formatted(fullDimension, numSubspaces));
        }
        this.fullDimension = fullDimension;
        this.numSubspaces = numSubspaces;
        this.numCentroids = numCentroids;
        this.subDim = fullDimension / numSubspaces;
        this.codebooks = new float[numSubspaces][numCentroids][subDim];
        this.encodedVectors = new ConcurrentHashMap<>();
        this.trained = false;
    }

    /**
     * Train the codebook using K-means on the provided training vectors.
     * Must be called before encode() or search().
     *
     * @param trainingVectors array of training vectors
     * @param maxIterations   maximum K-means iterations per subspace
     */
    public void train(float[][] trainingVectors, int maxIterations) {
        if (trainingVectors.length < numCentroids) {
            throw new IllegalArgumentException(
                "Need at least %d training vectors, got %d"
                    .formatted(numCentroids, trainingVectors.length));
        }

        Random rng = new Random(42); // deterministic

        for (int m = 0; m < numSubspaces; m++) {
            // Extract subvectors for this subspace
            float[][] subVectors = new float[trainingVectors.length][subDim];
            for (int v = 0; v < trainingVectors.length; v++) {
                System.arraycopy(trainingVectors[v], m * subDim, subVectors[v], 0, subDim);
            }

            // Initialize centroids using random selection
            float[][] centroids = new float[numCentroids][subDim];
            Set<Integer> selected = new HashSet<>();
            for (int c = 0; c < numCentroids; c++) {
                int idx;
                do { idx = rng.nextInt(trainingVectors.length); } while (selected.contains(idx));
                selected.add(idx);
                System.arraycopy(subVectors[idx], 0, centroids[c], 0, subDim);
            }

            // K-means iterations
            int[] assignments = new int[trainingVectors.length];
            for (int iter = 0; iter < maxIterations; iter++) {
                boolean changed = false;

                // Assignment step: assign each vector to nearest centroid
                for (int v = 0; v < trainingVectors.length; v++) {
                    int bestCentroid = 0;
                    float bestDist = Float.MAX_VALUE;
                    for (int c = 0; c < numCentroids; c++) {
                        float dist = squaredDistance(subVectors[v], centroids[c]);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestCentroid = c;
                        }
                    }
                    if (assignments[v] != bestCentroid) {
                        assignments[v] = bestCentroid;
                        changed = true;
                    }
                }

                if (!changed) break;

                // Update step: recompute centroids
                float[][] sums = new float[numCentroids][subDim];
                int[] counts = new int[numCentroids];
                for (int v = 0; v < trainingVectors.length; v++) {
                    int c = assignments[v];
                    counts[c]++;
                    for (int d = 0; d < subDim; d++) {
                        sums[c][d] += subVectors[v][d];
                    }
                }
                for (int c = 0; c < numCentroids; c++) {
                    if (counts[c] > 0) {
                        for (int d = 0; d < subDim; d++) {
                            centroids[c][d] = sums[c][d] / counts[c];
                        }
                    }
                }
            }

            codebooks[m] = centroids;
        }

        trained = true;
    }

    /**
     * Encode a vector into compressed representation.
     * Returns M bytes, each representing the nearest centroid in that subspace.
     */
    public byte[] encode(float[] vector) {
        if (!trained) throw new IllegalStateException("Codebook not trained yet");
        if (vector.length != fullDimension) {
            throw new IllegalArgumentException("Vector dimension mismatch");
        }

        byte[] codes = new byte[numSubspaces];
        for (int m = 0; m < numSubspaces; m++) {
            // Extract subvector
            float[] subVec = new float[subDim];
            System.arraycopy(vector, m * subDim, subVec, 0, subDim);

            // Find nearest centroid
            int bestCentroid = 0;
            float bestDist = Float.MAX_VALUE;
            for (int c = 0; c < numCentroids; c++) {
                float dist = squaredDistance(subVec, codebooks[m][c]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCentroid = c;
                }
            }
            codes[m] = (byte) bestCentroid;
        }
        return codes;
    }

    /**
     * Store an encoded vector with an ID.
     */
    public void store(long id, float[] vector) {
        encodedVectors.put(id, encode(vector));
    }

    /**
     * Compute approximate squared Euclidean distance between a query vector
     * and a stored encoded vector using Asymmetric Distance Computation (ADC).
     * 
     * ADC pre-computes a lookup table from the query to all centroids,
     * then sums the sub-distances. This is much faster than decompressing.
     *
     * Time: O(M) per distance (just M table lookups + additions)
     */
    public float approximateDistance(float[] query, long storedId) {
        if (!trained) throw new IllegalStateException("Codebook not trained yet");

        byte[] codes = encodedVectors.get(storedId);
        if (codes == null) return Float.MAX_VALUE;

        // Build lookup table: [subspace][centroid] → squared distance
        float[][] lookupTable = buildLookupTable(query);

        // Sum sub-distances
        float totalDist = 0.0f;
        for (int m = 0; m < numSubspaces; m++) {
            int centroidIdx = Byte.toUnsignedInt(codes[m]);
            totalDist += lookupTable[m][centroidIdx];
        }
        return totalDist;
    }

    /**
     * Search for the K nearest vectors to a query using ADC.
     * Uses a min-heap for efficient top-K selection.
     */
    public List<SearchResult> search(float[] query, int k) {
        if (!trained) throw new IllegalStateException("Codebook not trained yet");

        // Build lookup table once
        float[][] lookupTable = buildLookupTable(query);

        // Scan all encoded vectors
        record Scored(long id, float dist) {}
        PriorityQueue<Scored> heap = new PriorityQueue<>(
            Comparator.comparingDouble(Scored::dist).reversed());

        for (var entry : encodedVectors.entrySet()) {
            byte[] codes = entry.getValue();
            float dist = 0.0f;
            for (int m = 0; m < numSubspaces; m++) {
                int centroidIdx = Byte.toUnsignedInt(codes[m]);
                dist += lookupTable[m][centroidIdx];
            }

            if (heap.size() < k) {
                heap.offer(new Scored(entry.getKey(), dist));
            } else if (dist < heap.peek().dist) {
                heap.poll();
                heap.offer(new Scored(entry.getKey(), dist));
            }
        }

        return heap.stream()
            .sorted(Comparator.comparingDouble(Scored::dist))
            .map(s -> new SearchResult(s.id(), (float) Math.sqrt(s.dist())))
            .toList();
    }

    /**
     * Build the ADC lookup table for a query vector.
     * lookupTable[m][c] = squared distance between query subvector m and centroid c
     */
    private float[][] buildLookupTable(float[] query) {
        float[][] table = new float[numSubspaces][numCentroids];
        for (int m = 0; m < numSubspaces; m++) {
            float[] subQuery = new float[subDim];
            System.arraycopy(query, m * subDim, subQuery, 0, subDim);
            for (int c = 0; c < numCentroids; c++) {
                table[m][c] = squaredDistance(subQuery, codebooks[m][c]);
            }
        }
        return table;
    }

    /**
     * Squared Euclidean distance between two vectors.
     */
    private float squaredDistance(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * Get the compression ratio.
     */
    public double compressionRatio() {
        int originalBytes = fullDimension * Float.BYTES;
        int compressedBytes = numSubspaces; // 1 byte per subspace
        return (double) originalBytes / compressedBytes;
    }

    /**
     * Get the number of stored vectors.
     */
    public int size() {
        return encodedVectors.size();
    }

    /**
     * Memory usage in bytes for stored vectors only.
     */
    public long memoryUsage() {
        return (long) encodedVectors.size() * numSubspaces;
    }

    /**
     * Whether the codebook has been trained.
     */
    public boolean isTrained() {
        return trained;
    }

    /**
     * Search result record.
     */
    public record SearchResult(long id, float approximateDistance) {}
}
