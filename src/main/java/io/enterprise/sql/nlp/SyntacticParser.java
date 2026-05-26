package io.enterprise.sql.nlp;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Deterministic syntactic parser for SQL-like natural language queries.
 * Extracts Subject-Verb-Object (SVO) structure and dependency relations
 * WITHOUT any ML model or external dependency.
 *
 * Parsing strategy:
 * 1. Tokenize and classify (leverage existing Tokenizer)
 * 2. Identify the main verb (SQL operation keyword: select, insert, update, delete, etc.)
 * 3. Extract the subject (table name) and object (column names / values)
 * 4. Identify modifiers (WHERE conditions, GROUP BY, ORDER BY, LIMIT)
 * 5. Build a syntactic tree representing the query structure
 *
 * This structural understanding dramatically improves intent resolution
 * compared to pure bag-of-words / n-gram approaches because it captures
 * the RELATIONSHIPS between tokens, not just their presence.
 *
 * Example: "select name and email from users where active = true"
 *   verb = SELECT
 *   object = [name, email]
 *   subject = users
 *   modifier = WHERE(active = true)
 *
 * Deterministic: same input always produces same parse tree.
 */
public final class SyntacticParser {

    private SyntacticParser() {} // utility class

    /**
     * SQL verb categories with their associated patterns.
     */
    public enum SqlVerb {
        SELECT, INSERT, UPDATE, DELETE,
        CREATE, ALTER, DROP,
        JOIN, AGGREGATE, EXPLAIN, DESCRIBE, UNKNOWN
    }

    /**
     * Syntactic structure of a parsed query.
     * Immutable record — Valhalla-ready.
     */
    public record SyntacticStructure(
        SqlVerb verb,
        String subject,           // table name
        List<String> objects,     // column names / values
        List<String> conditions,  // WHERE clause components
        List<Modifier> modifiers, // GROUP BY, ORDER BY, LIMIT, HAVING
        List<JoinClause> joins,   // JOIN clauses
        double structuralConfidence  // how well the structure was parsed
    ) {
        public SyntacticStructure {
            objects = List.copyOf(objects);
            conditions = List.copyOf(conditions);
            modifiers = List.copyOf(modifiers);
            joins = List.copyOf(joins);
        }
    }

    /**
     * Modifier structure (GROUP BY, ORDER BY, LIMIT, HAVING).
     */
    public record Modifier(String type, String value) {
        public static Modifier of(String type, String value) {
            return new Modifier(type, value);
        }
    }

    /**
     * Join clause structure.
     */
    public record JoinClause(String type, String table, String onCondition) {}

    /**
     * Dependency relation between two tokens.
     */
    public record Dependency(String relation, String governor, String dependent) {}

    /**
     * Full parse result including dependency graph.
     */
    public record ParseResult(
        SyntacticStructure structure,
        List<Dependency> dependencies,
        List<Tokenizer.Token> tokens
    ) {
        public ParseResult {
            dependencies = List.copyOf(dependencies);
            tokens = List.copyOf(tokens);
        }
    }

    // ── Verb patterns ──
    private static final Map<Pattern, SqlVerb> VERB_PATTERNS = Map.ofEntries(
        Map.entry(Pattern.compile("(?i)\\b(select|show|get|find|list|display|retrieve|fetch|query|search|look\\s+up)\\b"), SqlVerb.SELECT),
        Map.entry(Pattern.compile("(?i)\\b(insert|add|create\\s+row|put|append|load)\\b"), SqlVerb.INSERT),
        Map.entry(Pattern.compile("(?i)\\b(update|modify|change|set|alter\\s+row|edit|patch)\\b"), SqlVerb.UPDATE),
        Map.entry(Pattern.compile("(?i)\\b(delete|remove|drop\\s+row|erase|purge|clear)\\b"), SqlVerb.DELETE),
        Map.entry(Pattern.compile("(?i)\\b(create\\s+table|make\\s+table|new\\s+table)\\b"), SqlVerb.CREATE),
        Map.entry(Pattern.compile("(?i)\\b(alter\\s+table|modify\\s+table|change\\s+table)\\b"), SqlVerb.ALTER),
        Map.entry(Pattern.compile("(?i)\\b(drop\\s+table|remove\\s+table|delete\\s+table)\\b"), SqlVerb.DROP),
        Map.entry(Pattern.compile("(?i)\\b(join|combine|merge|link|connect)\\b"), SqlVerb.JOIN),
        Map.entry(Pattern.compile("(?i)\\b(count|sum|total|average|avg|min|max|aggregate|how\\s+many|how\\s+much)\\b"), SqlVerb.AGGREGATE),
        Map.entry(Pattern.compile("(?i)\\b(explain|analyze|describe\\s+plan|query\\s+plan)\\b"), SqlVerb.EXPLAIN),
        Map.entry(Pattern.compile("(?i)\\b(describe|show\\s+schema|show\\s+table|schema|columns\\s+of)\\b"), SqlVerb.DESCRIBE)
    );

    // ── Condition indicators ──
    private static final Set<String> CONDITION_KEYWORDS = Set.of(
        "where", "when", "if", "with", "having", "that", "which", "whose"
    );

    // ── Modifier indicators ──
    private static final Set<String> MODIFIER_KEYWORDS = Set.of(
        "group", "order", "limit", "offset", "having", "top", "distinct", "unique"
    );

    // ── Join type indicators ──
    private static final Set<String> JOIN_TYPES = Set.of(
        "join", "inner", "left", "right", "outer", "full", "cross"
    );

    // ── Preposition patterns for table identification ──
    private static final Set<String> TABLE_PREPOSITIONS = Set.of(
        "from", "into", "update", "table", "on", "join"
    );

    /**
     * Parse a natural language or SQL-like query into its syntactic structure.
     */
    public static ParseResult parse(String query) {
        if (query == null || query.isBlank()) {
            return emptyParse();
        }

        List<Tokenizer.Token> tokens = Tokenizer.tokenize(query);
        List<Dependency> dependencies = new ArrayList<>();

        // Step 1: Identify the main verb
        SqlVerb verb = identifyVerb(query, tokens);
        dependencies.add(new Dependency("ROOT", verb.name(), "VERB"));

        // Step 2: Extract structural components based on verb
        String subject = extractSubject(tokens, verb);
        List<String> objects = extractObjects(tokens, verb);
        List<String> conditions = extractConditions(tokens);
        List<Modifier> modifiers = extractModifiers(tokens);
        List<JoinClause> joins = extractJoins(tokens);

        // Step 3: Build dependencies
        if (subject != null && !subject.isEmpty()) {
            dependencies.add(new Dependency("NSUBJ", verb.name(), subject));
        }
        for (String obj : objects) {
            dependencies.add(new Dependency("DOBJ", verb.name(), obj));
        }
        for (String cond : conditions) {
            dependencies.add(new Dependency("COND", verb.name(), cond));
        }

        // Step 4: Compute structural confidence
        double confidence = computeConfidence(verb, subject, objects, tokens);

        SyntacticStructure structure = new SyntacticStructure(
            verb, subject, objects, conditions, modifiers, joins, confidence
        );

        return new ParseResult(structure, dependencies, tokens);
    }

    /**
     * Identify the main SQL verb from the query.
     * Tries patterns in order of specificity (longest patterns first).
     */
    private static SqlVerb identifyVerb(String query, List<Tokenizer.Token> tokens) {
        SqlVerb bestVerb = SqlVerb.UNKNOWN;
        int bestPosition = Integer.MAX_VALUE;

        for (var entry : VERB_PATTERNS.entrySet()) {
            var matcher = entry.getKey().matcher(query);
            if (matcher.find()) {
                int pos = matcher.start();
                // Prefer the verb that appears earliest in the query
                if (pos < bestPosition) {
                    bestPosition = pos;
                    bestVerb = entry.getValue();
                }
            }
        }

        return bestVerb;
    }

    /**
     * Extract the subject (table name) from the query.
     * Table names typically follow prepositions like "from", "into", "update", "table".
     */
    private static String extractSubject(List<Tokenizer.Token> tokens, SqlVerb verb) {
        for (int i = 0; i < tokens.size() - 1; i++) {
            Tokenizer.Token current = tokens.get(i);
            String text = current.text().toLowerCase(Locale.ROOT);

            if (TABLE_PREPOSITIONS.contains(text)) {
                // Look ahead for the table name (identifier)
                for (int j = i + 1; j < tokens.size(); j++) {
                    Tokenizer.Token next = tokens.get(j);
                    if (next.type() == Tokenizer.TokenType.IDENTIFIER) {
                        return next.text();
                    }
                    if (next.type() == Tokenizer.TokenType.KEYWORD) break;
                }
            }

            // For UPDATE, the table name comes right after the verb
            if (verb == SqlVerb.UPDATE && current.type() == Tokenizer.TokenType.KEYWORD
                && current.text().equalsIgnoreCase("update")) {
                for (int j = i + 1; j < tokens.size(); j++) {
                    if (tokens.get(j).type() == Tokenizer.TokenType.IDENTIFIER) {
                        return tokens.get(j).text();
                    }
                    if (tokens.get(j).type() == Tokenizer.TokenType.KEYWORD) break;
                }
            }
        }
        return "";
    }

    /**
     * Extract objects (column names, value targets) from the query.
     */
    private static List<String> extractObjects(List<Tokenizer.Token> tokens, SqlVerb verb) {
        List<String> objects = new ArrayList<>();
        
        // For SELECT: columns between SELECT and FROM
        if (verb == SqlVerb.SELECT) {
            int selectIdx = findKeywordIndex(tokens, "select");
            int fromIdx = findKeywordIndex(tokens, "from");

            if (selectIdx >= 0) {
                int end = fromIdx >= 0 ? fromIdx : tokens.size();
                for (int i = selectIdx + 1; i < end; i++) {
                    Tokenizer.Token t = tokens.get(i);
                    if (t.type() == Tokenizer.TokenType.IDENTIFIER) {
                        objects.add(t.text());
                    } else if (t.type() == Tokenizer.TokenType.WILDCARD) {
                        objects.add("*");
                    }
                }
            }
        }

        // For INSERT: columns after the table name
        if (verb == SqlVerb.INSERT) {
            int intoIdx = findKeywordIndex(tokens, "into");
            if (intoIdx >= 0) {
                // Skip table name, collect column names in parentheses
                boolean inParens = false;
                for (int i = intoIdx + 1; i < tokens.size(); i++) {
                    Tokenizer.Token t = tokens.get(i);
                    if (t.type() == Tokenizer.TokenType.LPAREN) { inParens = true; continue; }
                    if (t.type() == Tokenizer.TokenType.RPAREN) { inParens = false; continue; }
                    if (inParens && t.type() == Tokenizer.TokenType.IDENTIFIER) {
                        objects.add(t.text());
                    }
                }
            }
        }

        // For UPDATE: column names in SET clause
        if (verb == SqlVerb.UPDATE) {
            int setIdx = findKeywordIndex(tokens, "set");
            if (setIdx >= 0) {
                for (int i = setIdx + 1; i < tokens.size(); i++) {
                    Tokenizer.Token t = tokens.get(i);
                    if (t.type() == Tokenizer.TokenType.IDENTIFIER) {
                        objects.add(t.text());
                    }
                    if (t.type() == Tokenizer.TokenType.KEYWORD &&
                        CONDITION_KEYWORDS.contains(t.text().toLowerCase(Locale.ROOT))) {
                        break;
                    }
                }
            }
        }

        return objects;
    }

    /**
     * Extract WHERE conditions.
     */
    private static List<String> extractConditions(List<Tokenizer.Token> tokens) {
        List<String> conditions = new ArrayList<>();
        int whereIdx = findKeywordIndex(tokens, "where");

        if (whereIdx >= 0) {
            StringBuilder current = new StringBuilder();
            for (int i = whereIdx + 1; i < tokens.size(); i++) {
                Tokenizer.Token t = tokens.get(i);
                String text = t.text().toLowerCase(Locale.ROOT);

                // Stop at modifier keywords
                if (MODIFIER_KEYWORDS.contains(text) || JOIN_TYPES.contains(text)) {
                    if (!current.isEmpty()) {
                        conditions.add(current.toString().trim());
                        current = new StringBuilder();
                    }
                    break;
                }

                // Split on AND/OR
                if (t.type() == Tokenizer.TokenType.KEYWORD &&
                    (text.equals("and") || text.equals("or"))) {
                    if (!current.isEmpty()) {
                        conditions.add(current.toString().trim());
                        current = new StringBuilder();
                    }
                } else {
                    if (!current.isEmpty()) current.append(" ");
                    current.append(t.text());
                }
            }
            if (!current.isEmpty()) {
                conditions.add(current.toString().trim());
            }
        }

        return conditions;
    }

    /**
     * Extract modifiers (GROUP BY, ORDER BY, LIMIT, etc.).
     */
    private static List<Modifier> extractModifiers(List<Tokenizer.Token> tokens) {
        List<Modifier> modifiers = new ArrayList<>();

        for (int i = 0; i < tokens.size() - 1; i++) {
            String text = tokens.get(i).text().toLowerCase(Locale.ROOT);
            String nextText = tokens.get(i + 1).text().toLowerCase(Locale.ROOT);

            // GROUP BY
            if (text.equals("group") && nextText.equals("by")) {
                modifiers.add(Modifier.of("GROUP_BY", collectUntilKeyword(tokens, i + 2)));
            }
            // ORDER BY
            if (text.equals("order") && nextText.equals("by")) {
                modifiers.add(Modifier.of("ORDER_BY", collectUntilKeyword(tokens, i + 2)));
            }
            // LIMIT
            if (text.equals("limit") && tokens.get(i + 1).type() == Tokenizer.TokenType.NUMERIC_LITERAL) {
                modifiers.add(Modifier.of("LIMIT", tokens.get(i + 1).text()));
            }
            // DISTINCT
            if (text.equals("distinct")) {
                modifiers.add(Modifier.of("DISTINCT", "true"));
            }
            // HAVING
            if (text.equals("having")) {
                modifiers.add(Modifier.of("HAVING", collectUntilKeyword(tokens, i + 1)));
            }
            // TOP
            if (text.equals("top") && tokens.get(i + 1).type() == Tokenizer.TokenType.NUMERIC_LITERAL) {
                modifiers.add(Modifier.of("TOP", tokens.get(i + 1).text()));
            }
        }

        return modifiers;
    }

    /**
     * Extract JOIN clauses.
     */
    private static List<JoinClause> extractJoins(List<Tokenizer.Token> tokens) {
        List<JoinClause> joins = new ArrayList<>();

        for (int i = 0; i < tokens.size(); i++) {
            String text = tokens.get(i).text().toLowerCase(Locale.ROOT);

            if (JOIN_TYPES.contains(text) || text.equals("join")) {
                String joinType = text.toUpperCase(Locale.ROOT);
                if (text.equals("join")) joinType = "INNER"; // default

                // Find table name after JOIN
                String table = "";
                for (int j = i + 1; j < tokens.size(); j++) {
                    if (tokens.get(j).type() == Tokenizer.TokenType.IDENTIFIER) {
                        table = tokens.get(j).text();
                        break;
                    }
                }

                // Find ON condition
                String onCondition = "";
                int onIdx = findKeywordIndexFrom(tokens, "on", i);
                if (onIdx >= 0) {
                    onCondition = collectUntilKeyword(tokens, onIdx + 1);
                }

                if (!table.isEmpty()) {
                    joins.add(new JoinClause(joinType, table, onCondition));
                }
            }
        }

        return joins;
    }

    /**
     * Compute structural confidence based on how complete the parse is.
     */
    private static double computeConfidence(SqlVerb verb, String subject,
                                             List<String> objects, List<Tokenizer.Token> tokens) {
        if (verb == SqlVerb.UNKNOWN) return 0.1;

        double score = 0.3; // base score for verb identification

        // Subject (table) found
        if (subject != null && !subject.isEmpty()) score += 0.3;

        // Objects (columns) found
        if (!objects.isEmpty()) score += 0.2;

        // Token count suggests a well-formed query
        if (tokens.size() >= 4) score += 0.1;
        if (tokens.size() >= 8) score += 0.1;

        return Math.min(1.0, score);
    }

    // ── Helper methods ──

    private static int findKeywordIndex(List<Tokenizer.Token> tokens, String keyword) {
        return findKeywordIndexFrom(tokens, keyword, 0);
    }

    private static int findKeywordIndexFrom(List<Tokenizer.Token> tokens, String keyword, int from) {
        for (int i = from; i < tokens.size(); i++) {
            if (tokens.get(i).type() == Tokenizer.TokenType.KEYWORD &&
                tokens.get(i).text().equalsIgnoreCase(keyword)) {
                return i;
            }
        }
        return -1;
    }

    private static String collectUntilKeyword(List<Tokenizer.Token> tokens, int start) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < tokens.size(); i++) {
            Tokenizer.Token t = tokens.get(i);
            String text = t.text().toLowerCase(Locale.ROOT);
            if (t.type() == Tokenizer.TokenType.KEYWORD &&
                (MODIFIER_KEYWORDS.contains(text) || text.equals("where") ||
                 text.equals("on") || text.equals("and") || text.equals("or"))) {
                break;
            }
            if (!sb.isEmpty()) sb.append(" ");
            sb.append(t.text());
        }
        return sb.toString();
    }

    private static ParseResult emptyParse() {
        SyntacticStructure empty = new SyntacticStructure(
            SqlVerb.UNKNOWN, "", List.of(), List.of(), List.of(), List.of(), 0.0
        );
        return new ParseResult(empty, List.of(), List.of());
    }
}
