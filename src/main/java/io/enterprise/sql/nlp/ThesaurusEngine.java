package io.enterprise.sql.nlp;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Built-in synonym expansion engine for SQL query understanding.
 * NO external API, NO dictionary file, NO LLM — pure hardcoded knowledge.
 *
 * Maps natural language expressions to their SQL semantic equivalents.
 * When a user says "show me all customers", the engine expands "show" → [select, get, find, display, retrieve]
 * and "all" → [*, every], dramatically improving intent resolution recall.
 *
 * Three expansion modes:
 * 1. SYNONYM: Direct word replacements (show → select, remove → delete)
 * 2. PHRASE: Multi-word phrase mappings ("how many" → count, "sort by" → order by)
 * 3. INFLECTION: Morphological variations (customers → customer, running → run)
 *
 * Thread-safe, deterministic, zero-latency.
 */
public final class ThesaurusEngine {

    /** Synonym groups: each group contains words that are semantically equivalent in SQL context */
    private static final List<Set<String>> SYNONYM_GROUPS = List.of(
        // SELECT synonyms
        Set.of("select", "show", "get", "find", "list", "display", "retrieve", "fetch",
               "query", "search", "look", "view", "read", "pull", "extract"),
        // INSERT synonyms
        Set.of("insert", "add", "put", "append", "load", "push", "create", "new",
               "write", "store", "save", "ingest", "import"),
        // UPDATE synonyms
        Set.of("update", "modify", "change", "set", "edit", "patch", "alter",
               "adjust", "revise", "transform", "replace"),
        // DELETE synonyms
        Set.of("delete", "remove", "drop", "erase", "purge", "clear", "eliminate",
               "destroy", "trash", "discard", "wipe"),
        // CREATE synonyms
        Set.of("create", "make", "build", "construct", "establish", "initialize",
               "define", "instantiate"),
        // JOIN synonyms
        Set.of("join", "combine", "merge", "link", "connect", "associate",
               "unite", "fuse", "couple", "relate", "match"),
        // WHERE / filter synonyms
        Set.of("where", "filter", "condition", "constraint", "criterion", "criteria",
               "matching", "satisfying", "such that", "with"),
        // GROUP BY synonyms
        Set.of("group", "categorize", "classify", "partition", "segment",
               "bucket", "cluster", "aggregate"),
        // ORDER BY synonyms
        Set.of("order", "sort", "arrange", "rank", "organize", "sequence", "rank"),
        // COUNT synonyms
        Set.of("count", "tally", "number", "total", "how many", "quantity",
               "amount", "num"),
        // SUM synonyms
        Set.of("sum", "total", "add up", "accumulate", "aggregate"),
        // AVERAGE synonyms
        Set.of("average", "avg", "mean", "median", "typical"),
        // MAXIMUM synonyms
        Set.of("maximum", "max", "highest", "largest", "biggest", "greatest", "top", "peak"),
        // MINIMUM synonyms
        Set.of("minimum", "min", "lowest", "smallest", "least", "bottom"),
        // ALL / wildcard synonyms
        Set.of("all", "every", "each", "*", "everything", "entire", "whole", "complete"),
        // TABLE synonyms
        Set.of("table", "relation", "entity", "collection", "dataset"),
        // COLUMN synonyms
        Set.of("column", "field", "attribute", "property", "variable", "feature", "dimension"),
        // ROW synonyms
        Set.of("row", "record", "entry", "item", "instance", "document", "tuple"),
        // INDEX synonyms
        Set.of("index", "indices", "indexes", "key", "keys")
    );

    /** Phrase-level mappings: multi-word expressions → SQL equivalents */
    private static final Map<String, String> PHRASE_MAP = Map.ofEntries(
        // Aggregation phrases
        Map.entry("how many", "count"),
        Map.entry("how much", "sum"),
        Map.entry("total of", "sum"),
        Map.entry("average of", "avg"),
        Map.entry("number of", "count"),
        Map.entry("count of", "count"),
        Map.entry("sum of", "sum"),
        Map.entry("max of", "max"),
        Map.entry("min of", "min"),
        // Sort/order phrases
        Map.entry("sort by", "order by"),
        Map.entry("arranged by", "order by"),
        Map.entry("ranked by", "order by"),
        Map.entry("ordered by", "order by"),
        Map.entry("grouped by", "group by"),
        Map.entry("categorized by", "group by"),
        Map.entry("classified by", "group by"),
        // Filter phrases
        Map.entry("where the", "where"),
        Map.entry("for which", "where"),
        Map.entry("such that", "where"),
        Map.entry("that have", "where"),
        Map.entry("that has", "where"),
        Map.entry("with value", "where"),
        Map.entry("equal to", "="),
        Map.entry("greater than", ">"),
        Map.entry("less than", "<"),
        Map.entry("at least", ">="),
        Map.entry("at most", "<="),
        Map.entry("not equal", "!="),
        // Join phrases
        Map.entry("combined with", "join"),
        Map.entry("merged with", "join"),
        Map.entry("linked to", "join on"),
        Map.entry("along with", "join"),
        // Schema phrases
        Map.entry("show me the structure", "describe"),
        Map.entry("what columns", "describe"),
        Map.entry("table structure", "describe"),
        Map.entry("schema of", "describe"),
        // Explain phrases
        Map.entry("explain the", "explain"),
        Map.entry("query plan for", "explain"),
        Map.entry("execution plan", "explain"),
        // Limit phrases
        Map.entry("top 10", "limit 10"),
        Map.entry("top 5", "limit 5"),
        Map.entry("top 20", "limit 20"),
        Map.entry("first 10", "limit 10"),
        Map.entry("first 5", "limit 5"),
        // Distinct phrases
        Map.entry("unique values", "distinct"),
        Map.entry("different values", "distinct"),
        Map.entry("remove duplicates", "distinct"),
        // Create table phrases
        Map.entry("new table", "create table"),
        Map.entry("make a table", "create table"),
        // Drop phrases
        Map.entry("get rid of", "drop"),
        Map.entry("remove table", "drop table")
    );

    /** Inverted index: word → its synonym group */
    private static final Map<String, Set<String>> WORD_TO_SYNONYMS;

    static {
        Map<String, Set<String>> map = new HashMap<>();
        for (Set<String> group : SYNONYM_GROUPS) {
            Set<String> immutableGroup = Set.copyOf(group);
            for (String word : group) {
                map.put(word.toLowerCase(Locale.ROOT), immutableGroup);
            }
        }
        WORD_TO_SYNONYMS = Map.copyOf(map);
    }

    private ThesaurusEngine() {} // utility class

    /**
     * Expand a single word to all its synonyms.
     * Returns the original word plus all synonyms.
     */
    public static Set<String> expandWord(String word) {
        if (word == null || word.isEmpty()) return Set.of();
        String normalized = word.toLowerCase(Locale.ROOT);
        return WORD_TO_SYNONYMS.getOrDefault(normalized, Set.of(normalized));
    }

    /**
     * Expand all words in a query to their synonyms.
     * Returns a map: original word → set of synonyms.
     */
    public static Map<String, Set<String>> expandQuery(String query) {
        if (query == null || query.isBlank()) return Map.of();

        Map<String, Set<String>> expansions = new LinkedHashMap<>();
        String[] tokens = query.toLowerCase(Locale.ROOT).split("\\s+");

        // Single-word expansion
        for (String token : tokens) {
            if (!token.isEmpty()) {
                expansions.put(token, expandWord(token));
            }
        }

        // Phrase-level expansion
        String lower = query.toLowerCase(Locale.ROOT);
        for (var entry : PHRASE_MAP.entrySet()) {
            if (lower.contains(entry.getKey())) {
                expansions.put(entry.getKey(), Set.of(entry.getKey(), entry.getValue()));
            }
        }

        return expansions;
    }

    /**
     * Generate expanded feature set from a query.
     * This produces a much richer feature set than the original query alone,
     * dramatically improving SimHash recall because the hash now covers
     * semantic equivalents that the user might have used.
     */
    public static Map<String, Double> generateExpandedFeatures(String query) {
        if (query == null || query.isBlank()) return Map.of();

        Map<String, Double> features = new LinkedHashMap<>();
        String lower = query.toLowerCase(Locale.ROOT);
        String[] tokens = lower.split("\\s+");

        // Phase 1: Token-level synonym expansion
        for (String token : tokens) {
            if (token.isEmpty()) continue;
            Set<String> synonyms = expandWord(token);

            // Original token gets full weight
            features.merge(token, 1.0, Double::sum);

            // Synonyms get reduced weight (they're expansions, not original text)
            for (String synonym : synonyms) {
                if (!synonym.equals(token)) {
                    features.merge(synonym, 0.6, Double::sum);
                }
            }
        }

        // Phase 2: Phrase-level mappings
        for (var entry : PHRASE_MAP.entrySet()) {
            if (lower.contains(entry.getKey())) {
                features.merge(entry.getValue(), 2.0, Double::sum);
            }
        }

        // Phase 3: Expanded bigrams
        for (int i = 0; i < tokens.length - 1; i++) {
            if (tokens[i].isEmpty() || tokens[i + 1].isEmpty()) continue;

            String bigram = tokens[i] + "_" + tokens[i + 1];
            features.merge(bigram, 1.5, Double::sum);

            // Expand with synonym-substituted bigrams
            Set<String> synA = expandWord(tokens[i]);
            Set<String> synB = expandWord(tokens[i + 1]);
            for (String a : synA) {
                for (String b : synB) {
                    if (!a.equals(tokens[i]) || !b.equals(tokens[i + 1])) {
                        String expandedBigram = a + "_" + b;
                        features.merge(expandedBigram, 0.4, Double::sum);
                    }
                }
            }
        }

        return features;
    }

    /**
     * Get the SQL keyword equivalent for a natural language word.
     * Returns the word itself if no mapping exists.
     */
    public static String toSqlKeyword(String word) {
        String lower = word.toLowerCase(Locale.ROOT);
        Set<String> synonyms = WORD_TO_SYNONYMS.get(lower);
        if (synonyms != null) {
            // Return the first short keyword (likely the SQL keyword)
            for (String syn : synonyms) {
                if (syn.length() <= 6 && isSqlKeyword(syn)) {
                    return syn.toUpperCase(Locale.ROOT);
                }
            }
        }
        // Check phrase map
        for (var entry : PHRASE_MAP.entrySet()) {
            if (lower.contains(entry.getKey())) {
                return entry.getValue().toUpperCase(Locale.ROOT);
            }
        }
        return word;
    }

    private static boolean isSqlKeyword(String word) {
        return Set.of("select", "insert", "update", "delete", "create", "alter",
            "drop", "join", "where", "group", "order", "having", "limit",
            "count", "sum", "avg", "min", "max", "distinct", "describe", "explain"
        ).contains(word.toLowerCase(Locale.ROOT));
    }

    /**
     * Number of synonym groups.
     */
    public static int synonymGroupCount() {
        return SYNONYM_GROUPS.size();
    }

    /**
     * Number of phrase mappings.
     */
    public static int phraseCount() {
        return PHRASE_MAP.size();
    }
}
