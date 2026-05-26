package io.enterprise.sql;

import io.enterprise.sql.config.AssistantConfig;
import io.enterprise.sql.jakarta.JakartaAIAdapter;
import io.enterprise.sql.jakarta.SqlAgent;
import io.enterprise.sql.model.*;
import io.enterprise.sql.sql.QueryValidator;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Enterprise SQL Assistant — Main Entry Point.
 *
 * Pure Java 26 + Jakarta Agentic AI + DOP + Panama + Valhalla
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────┐
 * │                    SqlAssistant                         │
 * │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
 *  │  │  CLI /   │  │ Jakarta  │  │  ScopedValue Config  │ │
 * │  │  REPL    │──│ AI Agent │──│  Context Propagation │ │
 * │  └──────────┘  └────┬─────┘  └──────────────────────┘ │
 * │                      │                                  │
 * │  ┌───────────────────┼──────────────────────────────┐  │
 * │  │           Intent Resolution Pipeline             │  │
 * │  │  ┌─────────┐ ┌──────────┐ ┌───────────────────┐ │  │
 * │  │  │ Intent  │ │ Semantic │ │  Regex Rule-Based │ │  │
 * │  │  │  Graph  │ │  Hasher  │ │     Fallback      │ │  │
 * │  │  └─────────┘ └──────────┘ └───────────────────┘ │  │
 * │  └──────────────────────────────────────────────────┘  │
 * │                      │                                  │
 * │  ┌───────────────────┼──────────────────────────────┐  │
 * │  │            SQL Generation Pipeline                │  │
 * │  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │  │
 * │  │  │  Pattern │ │  Schema  │ │  Query Validator │ │  │
 * │  │  │ Matching │ │ Resolver │ │                  │ │  │
 * │  │  └──────────┘ └──────────┘ └──────────────────┘ │  │
 * │  └──────────────────────────────────────────────────┘  │
 * │                      │                                  │
 * │  ┌───────────────────┼──────────────────────────────┐  │
 * │  │              Vector Memory                        │  │
 * │  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │  │
 * │  │  │  HNSW    │ │  Panama  │ │  Vector API     │ │  │
 * │  │  │  Index   │ │  Memory  │ │  (SIMD)         │ │  │
 * │  │  └──────────┘ └──────────┘ └──────────────────┘ │  │
 * │  └──────────────────────────────────────────────────┘  │
 * └─────────────────────────────────────────────────────────┘
 *
 * Features:
 * - Zero external LLM dependency
 * - Zero GPU requirement
 * - Ultra-low latency (deterministic pipeline)
 * - Memory efficient (off-heap Panama storage)
 * - SIMD accelerated (Vector API)
 * - Thread-safe (virtual threads + ScopedValue)
 * - Cloud native (stateless, configurable)
 */
public class SqlAssistant {

    private static final String BANNER = """
        ╔══════════════════════════════════════════════════════════╗
        ║       Enterprise SQL Assistant v1.0.0                   ║
        ║       Pure Java 26 + Jakarta AI + DOP + Panama         ║
        ║       No LLM • No GPU • No OpenAI • Deterministic      ║
        ╚══════════════════════════════════════════════════════════╝
        """;

    private static final String HELP_TEXT = """
        Commands:
          <query>       Translate natural language to SQL
          :explain      Explain the last generated SQL
          :schema       Show loaded schema information
          :metrics      Show agent performance metrics
          :similar      Find similar past queries
          :help         Show this help message
          :quit         Exit the assistant
        """;

    private final SqlAgent agent;
    private final QueryValidator validator;
    private final AssistantConfig config;
    private JakartaAIAdapter.AgentState state;
    private SqlQuery lastQuery;

    public SqlAssistant() {
        this(AssistantConfig.defaults());
    }

    public SqlAssistant(AssistantConfig config) {
        this.config = config;
        this.agent = new SqlAgent(config);
        this.validator = new QueryValidator();
        this.state = JakartaAIAdapter.AgentState.initial();
    }

    /**
     * Process a single query and return the response.
     */
    public JakartaAIAdapter.AgentResponse processQuery(String query) {
        var response = agent.query(query, state);
        state = response.newState();
        lastQuery = response.query();
        return response;
    }

    /**
     * Run the interactive REPL.
     */
    public void runRepl() {
        System.out.println(BANNER);
        System.out.println("  Vector dimension: " + config.vectorDimension());
        System.out.println("  HNSW M=" + config.hnswMaxConnections() +
            " efConstruction=" + config.hnswEfConstruction());
        System.out.println("  SIMD width: " + io.enterprise.sql.vector.VectorOps.simdWidth() + " floats/op");
        System.out.println("  Type :help for commands\n");

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        AtomicBoolean running = new AtomicBoolean(true);

        while (running.get()) {
            try {
                System.out.print("sql> ");
                System.out.flush();
                String line = reader.readLine();

                if (line == null) {
                    running.set(false);
                    break;
                }

                line = line.trim();
                if (line.isEmpty()) continue;

                // Handle commands
                if (line.startsWith(":")) {
                    handleCommand(line, running);
                    continue;
                }

                // Process query
                long start = System.nanoTime();
                try {
                    var response = processQuery(line);
                    long elapsed = (System.nanoTime() - start) / 1000; // microseconds

                    System.out.println("\n┌─ Intent: " + formatIntent(response.intent()));
                    System.out.println("├─ Confidence: " + String.format("%.2f%%", response.confidence() * 100));
                    System.out.println("├─ Latency: " + elapsed + " μs");
                    System.out.println("├─ Valid: " + (response.valid() ? "✓" : "✗ " + response.validationMessage()));
                    System.out.println("└─ SQL:");
                    System.out.println("   " + response.query().sql());
                    System.out.println();

                    if (!response.query().parameters().isEmpty()) {
                        System.out.println("   Parameters: " + response.query().parameters().keySet());
                    }

                    // Validate
                    var validation = validator.validate(response.query());
                    if (!validation.warnings().isEmpty()) {
                        System.out.println("   ⚠ Warnings:");
                        validation.warnings().forEach(w -> System.out.println("     - " + w));
                    }

                } catch (Exception e) {
                    System.err.println("Error: " + e.getMessage());
                }

            } catch (IOException e) {
                System.err.println("I/O error: " + e.getMessage());
                running.set(false);
            }
        }

        System.out.println("\nGoodbye!");
    }

    /**
     * Handle REPL commands.
     */
    private void handleCommand(String command, AtomicBoolean running) {
        switch (command.toLowerCase(Locale.ROOT)) {
            case ":quit", ":exit", ":q" -> {
                running.set(false);
            }
            case ":help", ":h", ":?" -> {
                System.out.println(HELP_TEXT);
            }
            case ":metrics", ":m" -> {
                var metrics = agent.getMetrics();
                System.out.println("\nAgent Metrics:");
                metrics.forEach((k, v) -> System.out.println("  " + k + ": " + v));
                System.out.println();
            }
            case ":explain", ":e" -> {
                if (lastQuery != null) {
                    System.out.println("\nLast query: " + lastQuery.sql());
                    System.out.println("Intent: " + formatIntent(lastQuery.intent()));
                    System.out.println("Parameters: " + lastQuery.parameters());
                    System.out.println("Estimated cost: " + lastQuery.estimatedCost());
                } else {
                    System.out.println("No previous query to explain.");
                }
                System.out.println();
            }
            case ":schema", ":s" -> {
                System.out.println("\nSchema information not available in REPL mode.");
                System.out.println("Register schemas programmatically via SchemaResolver.\n");
            }
            case ":similar" -> {
                if (lastQuery != null) {
                    var results = agent.findSimilar(lastQuery.sql(), 5);
                    System.out.println("\nSimilar queries:");
                    results.forEach(r ->
                        System.out.println("  ID=" + r.id() + " dist=" +
                            String.format("%.4f", r.distance())));
                } else {
                    System.out.println("No previous query to compare.");
                }
                System.out.println();
            }
            default -> {
                System.out.println("Unknown command: " + command);
                System.out.println("Type :help for available commands.");
            }
        }
    }

    /**
     * Format an intent for display.
     */
    private String formatIntent(SqlIntent intent) {
        return switch (intent) {
            case SqlIntent.Select s      -> "SELECT from " + s.tables();
            case SqlIntent.Insert i      -> "INSERT into " + i.table();
            case SqlIntent.Update u      -> "UPDATE " + u.table();
            case SqlIntent.Delete d      -> "DELETE from " + d.table();
            case SqlIntent.CreateTable c -> "CREATE TABLE " + c.table();
            case SqlIntent.AlterTable a  -> "ALTER TABLE " + a.table();
            case SqlIntent.DropTable dt  -> "DROP TABLE " + dt.table();
            case SqlIntent.Join j        -> "JOIN " + j.leftTable() + " ↔ " + j.rightTable();
            case SqlIntent.Aggregate ag  -> "AGGREGATE " + ag.function() + " on " + ag.table();
            case SqlIntent.Subquery sq   -> "SUBQUERY";
            case SqlIntent.Explain ex    -> "EXPLAIN";
            case SqlIntent.SchemaInfo si -> "SCHEMA INFO";
            case SqlIntent.Unknown uk    -> "UNKNOWN (" + uk.reason() + ")";
        };
    }

    /**
     * Programmatic API: process a query and return the SQL string.
     */
    public String translate(String query) {
        var response = processQuery(query);
        return response.query().sql();
    }

    /**
     * Programmatic API: process a query with full response details.
     */
    public Map<String, Object> translateDetailed(String query) {
        var response = processQuery(query);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("query", query);
        result.put("sql", response.query().sql());
        result.put("parameters", response.query().parameters());
        result.put("intent", formatIntent(response.intent()));
        result.put("confidence", response.confidence());
        result.put("valid", response.valid());
        result.put("latencyMicros", response.latencyNanos() / 1000);
        return result;
    }

    /**
     * Main entry point.
     */
    public static void main(String[] args) {
        var assistant = new SqlAssistant();

        if (args.length > 0) {
            // Non-interactive mode: process single query
            String query = String.join(" ", args);
            var result = assistant.translateDetailed(query);
            System.out.println(result);
        } else {
            // Interactive REPL mode
            assistant.runRepl();
        }
    }
}
