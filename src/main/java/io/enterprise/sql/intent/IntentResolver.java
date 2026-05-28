package io.enterprise.sql.intent;

import io.enterprise.sql.config.AssistantConfig;
import io.enterprise.sql.hashing.SemanticHasher;
import io.enterprise.sql.model.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Intent resolver that combines IntentGraph traversal with
 * SemanticHasher matching and rule-based fallback.
 *
 * Resolution pipeline:
 * 1. IntentGraph: semantic hash → graph traversal → intent type
 * 2. SemanticHasher: hash → closest registered pattern → intent type
 * 3. Regex rules: keyword pattern matching → intent type
 * 4. Fallback: Unknown intent
 *
 * Deterministic: no randomness, no ML inference.
 * Thread-safe: all state is immutable or concurrent.
 */
public class IntentResolver {

    private final IntentGraph intentGraph;
    private final SemanticHasher semanticHasher;
    private final List<RuleEntry> rules;

    /**
     * Rule entry for regex-based intent matching.
     */
    private record RuleEntry(Pattern pattern, String intentType, double confidence) {}

    public IntentResolver(IntentGraph intentGraph, SemanticHasher semanticHasher) {
        this.intentGraph = intentGraph;
        this.semanticHasher = semanticHasher;
        this.rules = buildDefaultRules();
    }

    /**
     * Resolve a natural language query to a SqlIntent.
     * This is the main entry point for intent resolution.
     */
    public SqlIntent resolve(String query) {
        String normalized = normalize(query);

        // Phase 1: Intent graph resolution
        IntentGraph.ResolutionResult graphResult = intentGraph.resolve(normalized);
        if (graphResult.isHighConfidence()) {
            return buildIntent(normalized, graphResult.matchedNode().intentType(),
                graphResult.confidence(), normalized);
        }

        // Phase 2: Semantic hasher
        SemanticHasher.MatchResult hashResult = semanticHasher.match(normalized);
        if (hashResult.confidence() >= 0.7) {
            return buildIntent(normalized, hashResult.intentType(),
                hashResult.confidence(), normalized);
        }

        // Phase 3: Regex rules
        for (RuleEntry rule : rules) {
            if (rule.pattern().matcher(normalized).find()) {
                return buildIntent(normalized, rule.intentType(),
                    rule.confidence(), normalized);
            }
        }

        // Phase 4: Fallback
        return new SqlIntent.Unknown(normalized, 0.1, "No matching intent found");
    }

    /**
     * Build a concrete SqlIntent from the resolved type.
     */
    private SqlIntent buildIntent(String rawQuery, String intentType,
                                   double confidence, String normalized) {
        return switch (intentType) {
            case "Select" -> resolveSelectIntent(rawQuery, confidence, normalized);
            case "Insert" -> resolveInsertIntent(rawQuery, confidence, normalized);
            case "Update" -> resolveUpdateIntent(rawQuery, confidence, normalized);
            case "Delete" -> resolveDeleteIntent(rawQuery, confidence, normalized);
            case "CreateTable" -> resolveCreateTableIntent(rawQuery, confidence, normalized);
            case "AlterTable" -> new SqlIntent.AlterTable(rawQuery, confidence, "", "ADD", null);
            case "DropTable" -> new SqlIntent.DropTable(rawQuery, confidence, "");
            case "Join" -> resolveJoinIntent(rawQuery, confidence, normalized);
            case "Aggregate" -> resolveAggregateIntent(rawQuery, confidence, normalized);
            case "Explain" -> new SqlIntent.Explain(rawQuery, confidence, normalized);
            case "SchemaInfo" -> new SqlIntent.SchemaInfo(rawQuery, confidence, "", "all");
            default -> new SqlIntent.Unknown(rawQuery, confidence, "Unrecognized: " + intentType);
        };
    }

    private SqlIntent resolveSelectIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        List<String> columns = extractColumns(normalized);
        String whereClause = extractWhere(normalized);
        String orderBy = extractOrderBy(normalized);
        int limit = extractLimit(normalized);
        return new SqlIntent.Select(rawQuery, confidence, tables, columns, whereClause, orderBy, limit);
    }

    private SqlIntent resolveInsertIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        String table = tables.isEmpty() ? "" : tables.getFirst();
        return new SqlIntent.Insert(rawQuery, confidence, table, List.of(), List.of());
    }

    private SqlIntent resolveUpdateIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        String table = tables.isEmpty() ? "" : tables.getFirst();
        String whereClause = extractWhere(normalized);
        return new SqlIntent.Update(rawQuery, confidence, table, Map.of(), whereClause);
    }

    private SqlIntent resolveDeleteIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        String table = tables.isEmpty() ? "" : tables.getFirst();
        String whereClause = extractWhere(normalized);
        return new SqlIntent.Delete(rawQuery, confidence, table, whereClause);
    }

    private SqlIntent resolveCreateTableIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        String table = tables.isEmpty() ? "" : tables.getFirst();
        return new SqlIntent.CreateTable(rawQuery, confidence, table, List.of());
    }

    private SqlIntent resolveJoinIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        String left = tables.size() > 0 ? tables.get(0) : "";
        String right = tables.size() > 1 ? tables.get(1) : "";
        return new SqlIntent.Join(rawQuery, confidence, left, right, "INNER", "");
    }

    private SqlIntent resolveAggregateIntent(String rawQuery, double confidence, String normalized) {
        List<String> tables = extractTables(normalized);
        String table = tables.isEmpty() ? "" : tables.getFirst();
        String function = "COUNT";
        if (normalized.contains("sum")) function = "SUM";
        else if (normalized.contains("avg")) function = "AVG";
        else if (normalized.contains("min")) function = "MIN";
        else if (normalized.contains("max")) function = "MAX";
        String groupBy = extractGroupBy(normalized);
        return new SqlIntent.Aggregate(rawQuery, confidence, function, "*", table, groupBy);
    }

    // Simple extraction helpers (deterministic, no ML)
    private List<String> extractTables(String normalized) {
        List<String> tables = new ArrayList<>();
        String[] tokens = normalized.split("\\s+");
        for (int i = 0; i < tokens.length - 1; i++) {
            if (tokens[i].equals("from") || tokens[i].equals("into") ||
                tokens[i].equals("update") || tokens[i].equals("table") ||
                tokens[i].equals("join")) {
                String table = tokens[i + 1].replaceAll("[,;]", "");
                if (!table.isEmpty() && !isKeyword(table)) {
                    tables.add(table);
                }
            }
        }
        return tables;
    }

    private List<String> extractColumns(String normalized) {
        if (normalized.contains("*")) return List.of("*");
        return List.of(); // would need schema-aware parsing for real extraction
    }

    private String extractWhere(String normalized) {
        int idx = normalized.indexOf("where");
        return idx >= 0 ? normalized.substring(idx + 5).trim() : "";
    }

    private String extractOrderBy(String normalized) {
        int idx = normalized.indexOf("order by");
        return idx >= 0 ? normalized.substring(idx + 8).trim() : "";
    }

    private int extractLimit(String normalized) {
        int idx = normalized.indexOf("limit");
        if (idx < 0) return 0;
        String after = normalized.substring(idx + 5).trim();
        String[] parts = after.split("\\s+");
        try { return Integer.parseInt(parts[0]); }
        catch (NumberFormatException e) { return 0; }
    }

    private String extractGroupBy(String normalized) {
        int idx = normalized.indexOf("group by");
        return idx >= 0 ? normalized.substring(idx + 8).trim() : "";
    }

    private boolean isKeyword(String token) {
        return Set.of("select", "from", "where", "insert", "update", "delete",
            "create", "alter", "drop", "join", "on", "and", "or", "not",
            "group", "order", "having", "limit", "offset", "as", "set",
            "values", "into", "inner", "left", "right", "outer", "all")
            .contains(token);
    }

    private String normalize(String query) {
        return query.toLowerCase(Locale.ROOT)
            .replaceAll("[^a-z0-9_*\\s=<>!,;()]", " ")
            .replaceAll("\\s+", " ")
            .trim();
    }

    /**
     * Build default regex rules for intent matching.
     */
    private List<RuleEntry> buildDefaultRules() {
        return List.of(
            new RuleEntry(Pattern.compile("\\bselect\\b.*\\bfrom\\b"), "Select", 0.9),
            new RuleEntry(Pattern.compile("\\binsert\\b.*\\binto\\b"), "Insert", 0.9),
            new RuleEntry(Pattern.compile("\\bupdate\\b.*\\bset\\b"), "Update", 0.9),
            new RuleEntry(Pattern.compile("\\bdelete\\b.*\\bfrom\\b"), "Delete", 0.9),
            new RuleEntry(Pattern.compile("\\bcreate\\s+table\\b"), "CreateTable", 0.95),
            new RuleEntry(Pattern.compile("\\balter\\s+table\\b"), "AlterTable", 0.95),
            new RuleEntry(Pattern.compile("\\bdrop\\s+table\\b"), "DropTable", 0.95),
            new RuleEntry(Pattern.compile("\\bjoin\\b"), "Join", 0.7),
            new RuleEntry(Pattern.compile("\\b(count|sum|avg|min|max)\\s*\\("), "Aggregate", 0.85),
            new RuleEntry(Pattern.compile("\\bexplain\\b"), "Explain", 0.9),
            new RuleEntry(Pattern.compile("\\b(show|describe|schema)\\b"), "SchemaInfo", 0.8)
        );
    }
}
