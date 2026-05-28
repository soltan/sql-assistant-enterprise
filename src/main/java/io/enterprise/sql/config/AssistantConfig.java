package io.enterprise.sql.config;

import java.util.Map;

/**
 * Configuration for the SQL Assistant using ScopedValue for implicit context propagation.
 * Replaces ThreadLocal with ScopedValue — structured, immutable, and inheritable by virtual threads.
 *
 * ScopedValue advantages over ThreadLocal:
 * - Immutable within scope (no accidental mutation)
 * - Automatically inherited by virtual threads (StructuredTaskScope)
 * - Bounded lifetime (cannot leak beyond scope)
 * - Zero overhead when not bound
 */
public final class AssistantConfig {

    /** ScopedValue for the current configuration context */
    public static final ScopedValue<AssistantConfig> CURRENT = ScopedValue.newInstance();

    // Configuration fields
    private final int vectorDimension;
    private final int hnswMaxConnections;
    private final int hnswEfConstruction;
    private final int hnswEfSearch;
    private final int semanticHashMaxDistance;
    private final double intentConfidenceThreshold;
    private final long vectorStoreCapacity;
    private final int maxQueryLength;
    private final boolean strictValidation;
    private final Map<String, String> schemaMapping;

    private AssistantConfig(Builder builder) {
        this.vectorDimension = builder.vectorDimension;
        this.hnswMaxConnections = builder.hnswMaxConnections;
        this.hnswEfConstruction = builder.hnswEfConstruction;
        this.hnswEfSearch = builder.hnswEfSearch;
        this.semanticHashMaxDistance = builder.semanticHashMaxDistance;
        this.intentConfidenceThreshold = builder.intentConfidenceThreshold;
        this.vectorStoreCapacity = builder.vectorStoreCapacity;
        this.maxQueryLength = builder.maxQueryLength;
        this.strictValidation = builder.strictValidation;
        this.schemaMapping = Map.copyOf(builder.schemaMapping);
    }

    // Getters
    public int vectorDimension() { return vectorDimension; }
    public int hnswMaxConnections() { return hnswMaxConnections; }
    public int hnswEfConstruction() { return hnswEfConstruction; }
    public int hnswEfSearch() { return hnswEfSearch; }
    public int semanticHashMaxDistance() { return semanticHashMaxDistance; }
    public double intentConfidenceThreshold() { return intentConfidenceThreshold; }
    public long vectorStoreCapacity() { return vectorStoreCapacity; }
    public int maxQueryLength() { return maxQueryLength; }
    public boolean strictValidation() { return strictValidation; }
    public Map<String, String> schemaMapping() { return schemaMapping; }

    /**
     * Run a task with this configuration bound as the current context.
     * The config is available via AssistantConfig.CURRENT.get() within the runnable.
     */
    public void runWith(Runnable task) {
        ScopedValue.where(CURRENT, this).run(task);
    }

    /**
     * Call a supplier with this configuration bound as the current context.
     */
    public <T> T callWith(ScopedValue.CallableOp<T, RuntimeException> supplier) {
        return ScopedValue.where(CURRENT, this).call(supplier);
    }

    /**
     * Get the current configuration from the ScopedValue context.
     * Throws if no configuration is bound.
     */
    public static AssistantConfig current() {
        return CURRENT.get();
    }

    /**
     * Check if a configuration is bound in the current scope.
     */
    public static boolean isBound() {
        return CURRENT.isBound();
    }

    /**
     * Create a default production configuration.
     */
    public static AssistantConfig defaults() {
        return new Builder().build();
    }

    /**
     * Create a development configuration with lower resource requirements.
     */
    public static AssistantConfig development() {
        return new Builder()
            .vectorDimension(64)
            .hnswMaxConnections(8)
            .hnswEfConstruction(50)
            .vectorStoreCapacity(10_000)
            .build();
    }

    @Override
    public String toString() {
        return "AssistantConfig{dim=%d, M=%d, efC=%d, efS=%d, capacity=%d}".formatted(
            vectorDimension, hnswMaxConnections, hnswEfConstruction,
            hnswEfSearch, vectorStoreCapacity);
    }

    /**
     * Builder for AssistantConfig.
     */
    public static class Builder {
        private int vectorDimension = 128;
        private int hnswMaxConnections = 16;
        private int hnswEfConstruction = 200;
        private int hnswEfSearch = 100;
        private int semanticHashMaxDistance = 12;
        private double intentConfidenceThreshold = 0.6;
        private long vectorStoreCapacity = 1_000_000;
        private int maxQueryLength = 4096;
        private boolean strictValidation = true;
        private Map<String, String> schemaMapping = Map.of();

        public Builder vectorDimension(int val) { this.vectorDimension = val; return this; }
        public Builder hnswMaxConnections(int val) { this.hnswMaxConnections = val; return this; }
        public Builder hnswEfConstruction(int val) { this.hnswEfConstruction = val; return this; }
        public Builder hnswEfSearch(int val) { this.hnswEfSearch = val; return this; }
        public Builder semanticHashMaxDistance(int val) { this.semanticHashMaxDistance = val; return this; }
        public Builder intentConfidenceThreshold(double val) { this.intentConfidenceThreshold = val; return this; }
        public Builder vectorStoreCapacity(long val) { this.vectorStoreCapacity = val; return this; }
        public Builder maxQueryLength(int val) { this.maxQueryLength = val; return this; }
        public Builder strictValidation(boolean val) { this.strictValidation = val; return this; }
        public Builder schemaMapping(Map<String, String> val) { this.schemaMapping = val; return this; }

        public AssistantConfig build() {
            return new AssistantConfig(this);
        }
    }
}
