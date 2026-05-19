package io.enterprise.sql.jakarta;

import io.enterprise.sql.config.AssistantConfig;
import io.enterprise.sql.hashing.SemanticHasher;
import io.enterprise.sql.intent.IntentGraph;
import io.enterprise.sql.intent.IntentResolver;
import io.enterprise.sql.model.*;
import io.enterprise.sql.sql.SqlGenerator;
import io.enterprise.sql.vector.HNSWIndex;

import java.util.*;
import java.util.concurrent.*;

/**
 * The SQL Agent — the top-level agent implementation.
 *
 * Implements the Jakarta Agentic AI concepts:
 * - Agent identity: deterministic SQL assistant
 * - Agent capabilities: intent resolution, SQL generation, validation
 * - Agent memory: HNSW vector memory for semantic recall
 * - Agent tools: SQL generator, schema resolver, query validator
 *
 * Cloud-native: stateless between requests, state carried via AgentState.
 * Scalable: all internal state is in ConcurrentHashMap or off-heap.
 * Thread-safe: safe for concurrent use from virtual threads.
 */
public class SqlAgent {

    private final JakartaAIAdapter adapter;
    private final HNSWIndex vectorMemory;
    private final SemanticHasher semanticHasher;
    private final AssistantConfig config;

    /**
     * Agent metrics — accumulated since startup.
     */
    private final ConcurrentHashMap<String, AtomicLong> metrics;

    public SqlAgent(AssistantConfig config) {
        this.config = config;
        this.vectorMemory = new HNSWIndex(
            config.vectorDimension(),
            config.hnswMaxConnections(),
            config.hnswEfConstruction()
        );
        this.semanticHasher = new SemanticHasher();
        registerDefaultPatterns();

        IntentGraph intentGraph = new IntentGraph.Builder().buildDefault();
        IntentResolver intentResolver = new IntentResolver(intentGraph, semanticHasher);
        SqlGenerator sqlGenerator = new SqlGenerator();

        this.adapter = new JakartaAIAdapter(intentResolver, sqlGenerator, config);
        this.metrics = new ConcurrentHashMap<>();
        initMetrics();
    }

    /**
     * Process a natural language query and return a SQL query.
     * This is the main entry point for the agent.
     *
     * @param query natural language query
     * @return agent response with generated SQL
     */
    public JakartaAIAdapter.AgentResponse query(String query) {
        return query(query, JakartaAIAdapter.AgentState.initial());
    }

    /**
     * Process a query with existing conversation state.
     */
    public JakartaAIAdapter.AgentResponse query(String query,
                                                  JakartaAIAdapter.AgentState state) {
        incrementMetric("total_queries");

        try {
            var response = adapter.execute(query, state);

            // Store the query in vector memory for semantic recall
            SemanticHash hash = semanticHasher.hash(query);
            // In production, we'd compute a proper embedding vector here
            float[] dummyVector = computeHashBasedVector(hash, config.vectorDimension());
            vectorMemory.insert(System.nanoTime(), dummyVector, hash);

            incrementMetric("successful_queries");
            metrics.get("total_latency_nanos").addAndGet(response.latencyNanos());

            return response;
        } catch (Exception e) {
            incrementMetric("failed_queries");
            throw e;
        }
    }

    /**
     * Find similar past queries using the HNSW vector memory.
     */
    public List<HNSWIndex.SearchResult> findSimilar(String query, int k) {
        SemanticHash hash = semanticHasher.hash(query);
        float[] vector = computeHashBasedVector(hash, config.vectorDimension());
        return vectorMemory.search(vector, k, config.hnswEfSearch());
    }

    /**
     * Get agent metrics.
     */
    public Map<String, Long> getMetrics() {
        Map<String, Long> result = new LinkedHashMap<>();
        metrics.forEach((key, counter) -> result.put(key, counter.get()));
        return result;
    }

    /**
     * Compute a deterministic vector from a semantic hash.
     * This is a placeholder — in production, use a proper embedding model.
     * The hash bits are expanded into a float vector with sinusoidal encoding.
     */
    private float[] computeHashBasedVector(SemanticHash hash, int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            // Use hash bits to seed a deterministic pseudo-random sequence
            long seed = hash.high() ^ hash.low() ^ (long) i * 0x9e3779b97f4a7c15L;
            // Simple LCG to generate deterministic float
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            vector[i] = ((seed >>> 33) % 1000) / 1000.0f;
        }
        return vector;
    }

    private void registerDefaultPatterns() {
        semanticHasher.registerPattern("select columns from table where condition", "Select");
        semanticHasher.registerPattern("select * from table", "Select");
        semanticHasher.registerPattern("select count from table group by", "Aggregate");
        semanticHasher.registerPattern("insert into table values data", "Insert");
        semanticHasher.registerPattern("update table set column value where condition", "Update");
        semanticHasher.registerPattern("delete from table where condition", "Delete");
        semanticHasher.registerPattern("create table with columns types", "CreateTable");
        semanticHasher.registerPattern("alter table add column", "AlterTable");
        semanticHasher.registerPattern("drop table", "DropTable");
        semanticHasher.registerPattern("select from table join table on condition", "Join");
        semanticHasher.registerPattern("explain query plan", "Explain");
        semanticHasher.registerPattern("show schema table columns", "SchemaInfo");
    }

    private void initMetrics() {
        metrics.put("total_queries", new AtomicLong(0));
        metrics.put("successful_queries", new AtomicLong(0));
        metrics.put("failed_queries", new AtomicLong(0));
        metrics.put("total_latency_nanos", new AtomicLong(0));
    }

    private void incrementMetric(String name) {
        metrics.computeIfAbsent(name, k -> new AtomicLong(0)).incrementAndGet();
    }
}
