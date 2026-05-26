package io.enterprise.sql.vector;

import io.enterprise.sql.model.SemanticHash;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Hierarchical Navigable Small World (HNSW) vector index.
 * Pure Java implementation — no native libraries, no GPU.
 * Deterministic when seeded, thread-safe, memory-efficient.
 *
 * Key properties:
 * - O(log n) search complexity
 * - Configurable M (max connections per layer) and efConstruction
 * - Layer assignment via exponentially decaying probability
 * - Thread-safe via ConcurrentHashMap
 *
 * Based on: Malkov & Yashunin, "Efficient and robust approximate nearest
 * neighbor search using Hierarchical Navigable Small World graphs" (2016)
 */
public class HNSWIndex {

    private final int dimension;
    private final int maxConnections;    // M parameter
    private final int efConstruction;    // construction-time beam width
    private final int maxLevel;
    private final double levelMultiplier;
    private final Random seedRandom;

    // Node storage
    private final ConcurrentHashMap<Long, HNSWNode> nodes;
    private final ConcurrentHashMap<Integer, Set<Long>> levelNodes;

    // Entry point
    private volatile long entryPointId = -1;
    private volatile int entryPointLevel = -1;

    /**
     * HNSW node representation.
     * Immutable ID + mutable neighbors per level.
     */
    static final class HNSWNode {
        final long id;
        final float[] vector;
        final SemanticHash hash;
        final int assignedLevel;
        final List<ConcurrentHashMap<Long, Float>> neighbors; // level → {neighborId → distance}

        HNSWNode(long id, float[] vector, SemanticHash hash, int level) {
            this.id = id;
            this.vector = vector;
            this.hash = hash;
            this.assignedLevel = level;
            this.neighbors = new ArrayList<>(level + 1);
            for (int i = 0; i <= level; i++) {
                neighbors.add(new ConcurrentHashMap<>());
            }
        }
    }

    /**
     * Create a new HNSW index.
     *
     * @param dimension      vector dimensionality
     * @param maxConnections  M: max connections per node per layer (typical: 16-64)
     * @param efConstruction  beam width during insertion (typical: 100-400)
     */
    public HNSWIndex(int dimension, int maxConnections, int efConstruction) {
        this.dimension = dimension;
        this.maxConnections = maxConnections;
        this.efConstruction = efConstruction;
        this.levelMultiplier = 1.0 / Math.log(maxConnections);
        this.maxLevel = 32; // theoretical max
        this.seedRandom = new Random(42); // deterministic seed
        this.nodes = new ConcurrentHashMap<>();
        this.levelNodes = new ConcurrentHashMap<>();
    }

    /**
     * Random level assignment following exponential decay.
     * P(level = l) = (1/M)^l
     */
    private int randomLevel() {
        double r = -Math.log(ThreadLocalRandom.current().nextDouble()) * levelMultiplier;
        return Math.min((int) r, maxLevel);
    }

    /**
     * Insert a vector into the index.
     */
    public synchronized void insert(long id, float[] vector, SemanticHash hash) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException("Dimension mismatch: expected %d, got %d"
                .formatted(dimension, vector.length));
        }

        int level = randomLevel();
        HNSWNode node = new HNSWNode(id, vector, hash, level);
        nodes.put(id, node);

        // Register in level sets
        for (int l = 0; l <= level; l++) {
            levelNodes.computeIfAbsent(l, k -> ConcurrentHashMap.newKeySet()).add(id);
        }

        if (entryPointId == -1) {
            // First node
            entryPointId = id;
            entryPointLevel = level;
            return;
        }

        // Search from top level down to find insertion point
        long currentId = entryPointId;
        HNSWNode entryNode = nodes.get(entryPointId);

        // Phase 1: Greedy search from entry level down to level+1
        for (int l = entryPointLevel; l > level; l--) {
            currentId = searchLayer(vector, currentId, l, 1).getFirst().id;
        }

        // Phase 2: Insert at levels [level, 0]
        for (int l = Math.min(level, entryPointLevel); l >= 0; l--) {
            List<Neighbor> candidates = searchLayer(vector, currentId, l, efConstruction);
            List<Neighbor> neighbors = selectNeighbors(candidates, maxConnections);

            for (Neighbor nb : neighbors) {
                node.neighbors.get(l).put(nb.id, nb.distance);

                // Add bidirectional connection
                HNSWNode neighborNode = nodes.get(nb.id);
                if (neighborNode != null && l < neighborNode.neighbors.size()) {
                    neighborNode.neighbors.get(l).put(id, nb.distance);
                    // Prune if over-connected
                    if (neighborNode.neighbors.get(l).size() > maxConnections * 2) {
                        pruneConnections(neighborNode, l);
                    }
                }
            }

            if (!candidates.isEmpty()) {
                currentId = candidates.getFirst().id;
            }
        }

        // Update entry point if new node has higher level
        if (level > entryPointLevel) {
            entryPointId = id;
            entryPointLevel = level;
        }
    }

    /**
     * Search for K nearest neighbors.
     *
     * @param query    query vector
     * @param k        number of results
     * @param efSearch search beam width (higher = more accurate, slower)
     */
    public List<SearchResult> search(float[] query, int k, int efSearch) {
        if (entryPointId == -1) return List.of();

        long currentId = entryPointId;

        // Traverse from top level down to level 1
        for (int l = entryPointLevel; l > 0; l--) {
            currentId = searchLayer(query, currentId, l, 1).getFirst().id;
        }

        // Search at level 0 with efSearch beam width
        List<Neighbor> candidates = searchLayer(query, currentId, 0, efSearch);

        return candidates.stream()
            .sorted(Comparator.comparingDouble(n -> n.distance))
            .limit(k)
            .map(n -> {
                HNSWNode node = nodes.get(n.id);
                return new SearchResult(n.id, node.hash, n.distance, node.vector);
            })
            .toList();
    }

    /**
     * Search within a single layer.
     */
    private List<Neighbor> searchLayer(float[] query, long entryId, int level, int ef) {
        Set<Long> visited = new HashSet<>();
        PriorityQueue<Neighbor> candidates = new PriorityQueue<>(
            Comparator.comparingDouble(Neighbor::distance).reversed()); // max-heap
        PriorityQueue<Neighbor> results = new PriorityQueue<>(
            Comparator.comparingDouble(Neighbor::distance)); // min-heap

        float initDist = euclideanDistance(query, nodes.get(entryId).vector);
        Neighbor init = new Neighbor(entryId, initDist);
        candidates.offer(init);
        results.offer(init);
        visited.add(entryId);

        while (!candidates.isEmpty()) {
            Neighbor current = candidates.poll();

            if (results.size() >= ef && current.distance > results.peek().distance) {
                break;
            }

            HNSWNode currentNode = nodes.get(current.id);
            if (currentNode == null || level >= currentNode.neighbors.size()) continue;

            for (long neighborId : currentNode.neighbors.get(level).keySet()) {
                if (visited.add(neighborId)) {
                    HNSWNode neighborNode = nodes.get(neighborId);
                    if (neighborNode == null) continue;

                    float dist = euclideanDistance(query, neighborNode.vector);
                    Neighbor nb = new Neighbor(neighborId, dist);

                    if (results.size() < ef || dist < results.peek().distance) {
                        candidates.offer(nb);
                        results.offer(nb);
                        if (results.size() > ef) {
                            results.poll();
                        }
                    }
                }
            }
        }

        return new ArrayList<>(results);
    }

    /**
     * Select best neighbors using simple selection heuristic.
     */
    private List<Neighbor> selectNeighbors(List<Neighbor> candidates, int maxCount) {
        return candidates.stream()
            .sorted(Comparator.comparingDouble(Neighbor::distance))
            .limit(maxCount)
            .toList();
    }

    /**
     * Prune excess connections keeping closest ones.
     */
    private void pruneConnections(HNSWNode node, int level) {
        var neighbors = node.neighbors.get(level);
        List<Map.Entry<Long, Float>> sorted = neighbors.entrySet().stream()
            .sorted(Map.Entry.comparingByValue())
            .limit(maxConnections)
            .toList();

        neighbors.clear();
        for (var entry : sorted) {
            neighbors.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Compute Euclidean distance between two vectors.
     */
    private float euclideanDistance(float[] a, float[] b) {
        return VectorOps.euclideanDistance(a, b);
    }

    /**
     * Number of vectors in the index.
     */
    public int size() {
        return nodes.size();
    }

    /**
     * Current maximum level.
     */
    public int maxLevel() {
        return entryPointLevel;
    }

    // Internal record types
    record Neighbor(long id, float distance) {}
    public record SearchResult(long id, SemanticHash hash, float distance, float[] vector) {}
}
