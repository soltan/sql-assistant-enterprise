package io.enterprise.sql.jakarta;

import io.enterprise.sql.config.AssistantConfig;
import io.enterprise.sql.intent.IntentGraph;
import io.enterprise.sql.intent.IntentResolver;
import io.enterprise.sql.hashing.SemanticHasher;
import io.enterprise.sql.model.*;
import io.enterprise.sql.sql.SqlGenerator;

import java.util.*;
import java.util.concurrent.StructuredTaskScope;

/**
 * Jakarta Agentic AI API adapter for the SQL Assistant.
 * Implements the agentic loop: perceive → reason → act → reflect.
 *
 * This adapter bridges the Jakarta AI API concepts with our
 * deterministic SQL assistant pipeline:
 *
 * - Agent: the SQL Assistant itself
 * - Tool: SQL generation, schema resolution, query validation
 * - Memory: HNSW vector memory + semantic hash index
 * - Loop: single-pass deterministic (no iterative refinement needed)
 *
 * The agentic loop is:
 * 1. PERCEIVE: Receive natural language query
 * 2. REASON:   Resolve intent via IntentGraph + SemanticHasher
 * 3. ACT:      Generate SQL from resolved intent
 * 4. REFLECT:  Validate and optionally correct the SQL
 *
 * All steps are deterministic — no LLM, no randomness, no GPU.
 */
public class JakartaAIAdapter {

    private final IntentResolver intentResolver;
    private final SqlGenerator sqlGenerator;
    private final AssistantConfig config;

    /**
     * Agent state — tracks the conversation and accumulated context.
     * Immutable per interaction; new state is created for each turn.
     */
    public record AgentState(
        List<Interaction> history,
        Map<String, Object> context,
        long interactionCount
    ) {
        public AgentState {
            history = List.copyOf(history);
            context = Map.copyOf(context);
        }

        public static AgentState initial() {
            return new AgentState(List.of(), Map.of(), 0);
        }

        public AgentState withInteraction(Interaction interaction) {
            var newHistory = new ArrayList<>(history);
            newHistory.add(interaction);
            return new AgentState(newHistory, context, interactionCount + 1);
        }
    }

    /**
     * Single interaction record.
     */
    public record Interaction(
        String input,
        SqlIntent intent,
        SqlQuery query,
        boolean valid,
        String validationMessage,
        long latencyNanos
    ) {}

    /**
     * Agent response.
     */
    public record AgentResponse(
        SqlQuery query,
        SqlIntent intent,
        boolean valid,
        String validationMessage,
        long latencyNanos,
        AgentState newState
    ) {}

    public JakartaAIAdapter(IntentResolver intentResolver, SqlGenerator sqlGenerator,
                             AssistantConfig config) {
        this.intentResolver = intentResolver;
        this.sqlGenerator = sqlGenerator;
        this.config = config;
    }

    /**
     * Execute the agentic loop for a single query.
     * Uses ScopedValue for configuration context and StructuredTaskScope
     * for concurrent sub-task execution.
     */
    public AgentResponse execute(String query, AgentState state) {
        return config.callWith(() -> {
            long start = System.nanoTime();

            // Step 1: PERCEIVE — normalize input
            String normalizedQuery = perceive(query);

            // Step 2: REASON — resolve intent (potentially concurrent)
            SqlIntent intent;
            try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
                StructuredTaskScope.Subtask<SqlIntent> intentTask =
                    scope.fork(() -> reason(normalizedQuery));
                scope.join();
                scope.throwIfFailed();
                intent = intentTask.get();
            } catch (Exception e) {
                intent = new SqlIntent.Unknown(normalizedQuery, 0.0,
                    "Intent resolution failed: " + e.getMessage());
            }

            // Step 3: ACT — generate SQL
            SqlQuery sqlQuery = act(intent);

            // Step 4: REFLECT — validate
            var validation = reflect(sqlQuery);

            long latency = System.nanoTime() - start;

            Interaction interaction = new Interaction(
                query, intent, sqlQuery,
                validation.valid(), validation.message(), latency
            );

            AgentState newState = state.withInteraction(interaction);

            return new AgentResponse(
                sqlQuery, intent,
                validation.valid(), validation.message(),
                latency, newState
            );
        });
    }

    /**
     * PERCEIVE: Preprocess and normalize the input query.
     */
    private String perceive(String query) {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("Query must not be blank");
        }
        if (query.length() > config.maxQueryLength()) {
            throw new IllegalArgumentException(
                "Query exceeds max length: %d > %d".formatted(query.length(), config.maxQueryLength()));
        }
        return query.trim();
    }

    /**
     * REASON: Resolve the intent of the query.
     */
    private SqlIntent reason(String query) {
        return intentResolver.resolve(query);
    }

    /**
     * ACT: Generate SQL from the resolved intent.
     */
    private SqlQuery act(SqlIntent intent) {
        return sqlGenerator.generate(intent);
    }

    /**
     * REFLECT: Validate the generated SQL.
     */
    private ValidationResult reflect(SqlQuery query) {
        // Basic validation checks
        if (query.sql().isEmpty()) {
            return new ValidationResult(false, "Generated SQL is empty");
        }
        if (query.sql().length() > 65536) {
            return new ValidationResult(false, "Generated SQL exceeds 64KB");
        }
        // Check for basic SQL injection patterns
        String lower = query.sql().toLowerCase(Locale.ROOT);
        if (lower.contains(";") && lower.indexOf(';') < lower.length() - 1) {
            return new ValidationResult(false, "Potential SQL injection: multiple statements detected");
        }
        return new ValidationResult(true, "OK");
    }

    record ValidationResult(boolean valid, String message) {}
}
