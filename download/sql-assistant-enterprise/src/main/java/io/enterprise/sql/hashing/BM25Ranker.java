package io.enterprise.sql.hashing;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Pure Java BM25 (Okapi BM25) implementation — no external dependencies.
 *
 * BM25 is a probabilistic ranking function used in information retrieval to
 * estimate the relevance of documents to a given search query. It is based on
 * the probabilistic retrieval framework developed by Stephen Robertson and
 * Karen Sparck Jones in the 1970s-90s.
 *
 * BM25 scoring formula:
 *   score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
 *
 * Where:
 *   f(qi,D)  = term frequency of qi in document D
 *   |D|      = document length (term count)
 *   avgdl    = average document length across corpus
 *   k1       = term frequency saturation parameter (1.2-2.0 typical, default 1.5)
 *   b        = length normalization parameter (0.75 typical)
 *   IDF(qi)  = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
 *
 * Key advantages over simple term matching:
 *   1. IDF weighting: rare terms get higher scores (high discrimination)
 *   2. TF saturation: term frequency is non-linear (diminishing returns)
 *   3. Length normalization: penalizes overly long documents
 *   4. Probabilistic foundation: grounded in Robertson-Sparck Jones model
 *
 * In this SQL Assistant, BM25 serves 3 roles:
 *   1. Rank SQL templates by relevance to user query (template pattern + description as document)
 *   2. Rank tables/columns by relevance using COMMENT ON metadata
 *   3. Strategy in ensemble resolver (bm25_relevance, weight=1.4)
 *
 * Bilingual: French + English stop words are filtered for better IDF discrimination.
 * Accent normalization: é→e, è→e, ç→c etc. for cross-lingual matching.
 *
 * Thread-safe: all internal data structures use ConcurrentHashMap.
 */
public final class BM25Ranker {

    /** Term frequency saturation parameter (typical range: 1.2-2.0) */
    private final double k1;

    /** Length normalization parameter (typical range: 0.5-1.0) */
    private final double b;

    /** Bilingual stop words (English + French) — filtered for better IDF discrimination */
    private static final Set<String> STOP_WORDS = Set.of(
        // English stop words
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "where", "how", "what", "which",
        "who", "whom", "this", "that", "these", "those", "it", "its",
        "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their",
        // French stop words
        "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
        "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi",
        "dont", "est", "sont", "etait", "etaient", "a", "ont", "fait",
        "dans", "sur", "sous", "avec", "pour", "par", "en", "vers",
        "chez", "sans", "entre", "ce", "cet", "cette",
        "ces", "il", "elle", "ils", "elles", "nous", "vous", "on",
        "je", "tu", "me", "te", "se", "lui", "leur", "y",
        "ne", "pas", "plus", "aussi", "tres", "bien", "tout", "tous",
        "toute", "toutes", "autre", "autres", "meme", "chaque"
    );

    /** Document entry for BM25 index */
    public record BM25Document(String id, String intent, List<String> tokens, String source) {
        /** Document length in tokens */
        public int length() { return tokens.size(); }
    }

    /** Scored result */
    public record BM25Result(String id, String intent, double score, String source)
        implements Comparable<BM25Result> {
        @Override
        public int compareTo(BM25Result other) {
            return Double.compare(other.score, this.score); // descending
        }
    }

    /** Indexed documents */
    private final ConcurrentHashMap<String, BM25Document> documents;

    /** Document frequency per term: how many documents contain this term */
    private final ConcurrentHashMap<String, Integer> df;

    /** Total number of documents in the corpus */
    private volatile int totalDocs;

    /** Average document length across the corpus */
    private volatile double avgDocLength;

    /**
     * Create a BM25 ranker with default parameters (k1=1.5, b=0.75).
     */
    public BM25Ranker() {
        this(1.5, 0.75);
    }

    /**
     * Create a BM25 ranker with custom parameters.
     *
     * @param k1 term frequency saturation (1.2-2.0 typical)
     * @param b  length normalization (0.5-1.0 typical, 0.75 standard)
     */
    public BM25Ranker(double k1, double b) {
        this.k1 = k1;
        this.b = b;
        this.documents = new ConcurrentHashMap<>();
        this.df = new ConcurrentHashMap<>();
        this.totalDocs = 0;
        this.avgDocLength = 0;
    }

    // ═══════════════════════════════════════════════════════════
    // TOKENIZATION
    // ═══════════════════════════════════════════════════════════

    /**
     * Tokenize text for BM25 indexing/querying.
     * - Lowercase
     * - Normalize accents (é→e, è→e, ç→c, etc.)
     * - Split on non-alphanumeric characters
     * - Remove stop words (bilingual English + French)
     * - Discard single-character tokens
     *
     * @param text input text
     * @return list of clean tokens
     */
    public static List<String> tokenize(String text) {
        if (text == null || text.isBlank()) return List.of();

        // Normalize: lowercase + accent removal for cross-lingual matching
        String normalized = text.toLowerCase()
            .replace("\u00e9", "e").replace("\u00e8", "e").replace("\u00ea", "e").replace("\u00eb", "e")
            .replace("\u00e0", "a").replace("\u00e2", "a")
            .replace("\u00f9", "u").replace("\u00fb", "u")
            .replace("\u00f4", "o").replace("\u00ee", "i").replace("\u00ef", "i")
            .replace("\u00e7", "c")
            .replace("\u00e6", "ae").replace("\u0153", "oe");

        String[] raw = normalized.split("[^a-z0-9]+");
        List<String> tokens = new ArrayList<>();
        for (String t : raw) {
            if (t.length() > 1 && !STOP_WORDS.contains(t)) {
                tokens.add(t);
            }
        }
        return tokens;
    }

    // ═══════════════════════════════════════════════════════════
    // INDEXING
    // ═══════════════════════════════════════════════════════════

    /**
     * Add a document to the BM25 index.
     *
     * @param id     unique document identifier
     * @param intent associated intent type (e.g., "Select", "Aggregate")
     * @param text   document text content
     * @param source document source (e.g., "template", "schema_table")
     */
    public void addDocument(String id, String intent, String text, String source) {
        List<String> tokens = tokenize(text);
        BM25Document doc = new BM25Document(id, intent, tokens, source);
        documents.put(id, doc);

        // Update document frequencies (only unique terms per document)
        Set<String> uniqueTerms = new HashSet<>(tokens);
        for (String term : uniqueTerms) {
            df.merge(term, 1, Integer::sum);
        }

        // Update corpus statistics
        recalculateStats();
    }

    /**
     * Add a pre-tokenized document to the BM25 index.
     *
     * @param id     unique document identifier
     * @param intent associated intent type
     * @param tokens pre-tokenized content
     * @param source document source
     */
    public void addDocument(String id, String intent, List<String> tokens, String source) {
        BM25Document doc = new BM25Document(id, intent, tokens, source);
        documents.put(id, doc);

        Set<String> uniqueTerms = new HashSet<>(tokens);
        for (String term : uniqueTerms) {
            df.merge(term, 1, Integer::sum);
        }

        recalculateStats();
    }

    /**
     * Batch-add documents and rebuild stats once (more efficient than addDocument for bulk loading).
     *
     * @param batch list of document entries to add
     */
    public void addDocuments(List<BM25Document> batch) {
        for (BM25Document doc : batch) {
            documents.put(doc.id(), doc);
        }
        rebuildStats();
    }

    /**
     * Rebuild corpus statistics from scratch after batch operations.
     * Clears and recomputes df map, totalDocs, avgDocLength.
     */
    public void rebuildStats() {
        df.clear();
        long totalLen = 0;

        for (BM25Document doc : documents.values()) {
            totalLen += doc.length();
            Set<String> uniqueTerms = new HashSet<>(doc.tokens());
            for (String term : uniqueTerms) {
                df.merge(term, 1, Integer::sum);
            }
        }

        totalDocs = documents.size();
        avgDocLength = totalDocs > 0 ? (double) totalLen / totalDocs : 0;
    }

    /**
     * Recalculate stats incrementally (called after single document add).
     */
    private void recalculateStats() {
        totalDocs = documents.size();
        long totalLen = documents.values().stream()
            .mapToLong(BM25Document::length)
            .sum();
        avgDocLength = totalDocs > 0 ? (double) totalLen / totalDocs : 0;
    }

    /**
     * Remove a document from the index.
     *
     * @param id document identifier to remove
     */
    public void removeDocument(String id) {
        BM25Document removed = documents.remove(id);
        if (removed != null) {
            // Need to rebuild df since we can't easily decrement
            rebuildStats();
        }
    }

    /**
     * Clear all documents and reset the index.
     */
    public void clear() {
        documents.clear();
        df.clear();
        totalDocs = 0;
        avgDocLength = 0;
    }

    // ═══════════════════════════════════════════════════════════
    // SCORING
    // ═══════════════════════════════════════════════════════════

    /**
     * Compute IDF (Inverse Document Frequency) for a term.
     * Uses the standard BM25 IDF formula:
     *   IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
     *
     * Where N = total documents, n(t) = documents containing term t.
     * The +1 inside ln ensures IDF is always non-negative.
     *
     * @param term the term to compute IDF for
     * @return IDF value (higher = rarer term = more discriminating)
     */
    public double idf(String term) {
        int n = df.getOrDefault(term, 0);
        return Math.log((totalDocs - n + 0.5) / (n + 0.5) + 1.0);
    }

    /**
     * Compute BM25 score of a query against a single document.
     *
     * @param queryTokens tokenized query
     * @param doc         document to score against
     * @return BM25 relevance score (non-negative)
     */
    public double score(List<String> queryTokens, BM25Document doc) {
        if (avgDocLength == 0 || queryTokens.isEmpty()) return 0;

        double score = 0;
        int docLen = doc.length();

        // Build term frequency map for the document
        Map<String, Long> tfMap = doc.tokens().stream()
            .collect(Collectors.groupingBy(t -> t, Collectors.counting()));

        for (String qTerm : queryTokens) {
            long tf = tfMap.getOrDefault(qTerm, 0L);
            if (tf == 0) continue; // term not in document, no contribution

            double idfVal = idf(qTerm);
            double numerator = tf * (k1 + 1);
            double denominator = tf + k1 * (1 - b + b * docLen / avgDocLength);
            score += idfVal * numerator / denominator;
        }

        return score;
    }

    /**
     * Compute BM25 score of a query string against a document.
     *
     * @param query query string (will be tokenized)
     * @param doc   document to score against
     * @return BM25 relevance score
     */
    public double score(String query, BM25Document doc) {
        return score(tokenize(query), doc);
    }

    /**
     * Rank all documents against a query, return top-K results.
     *
     * @param query query string
     * @param topK  maximum number of results
     * @return list of scored results, sorted by descending score
     */
    public List<BM25Result> rank(String query, int topK) {
        List<String> queryTokens = tokenize(query);
        if (queryTokens.isEmpty() || documents.isEmpty()) return List.of();

        List<BM25Result> results = new ArrayList<>();
        for (BM25Document doc : documents.values()) {
            double s = score(queryTokens, doc);
            if (s > 0) {
                results.add(new BM25Result(doc.id, doc.intent, s, doc.source));
            }
        }

        results.sort(Comparator.comparingDouble(BM25Result::score).reversed());
        return results.size() > topK ? results.subList(0, topK) : results;
    }

    /**
     * Rank documents and return normalized scores (0-1 range).
     * Top result always gets score 1.0, others are relative.
     *
     * @param query query string
     * @param topK  maximum number of results
     * @return list of scored results with normalized scores
     */
    public List<BM25Result> rankNormalized(String query, int topK) {
        List<BM25Result> raw = rank(query, topK);
        if (raw.isEmpty()) return raw;

        double maxScore = raw.get(0).score;
        if (maxScore <= 0) return raw;

        return raw.stream()
            .map(r -> new BM25Result(r.id, r.intent, r.score / maxScore, r.source))
            .collect(Collectors.toList());
    }

    /**
     * Find the best matching intent via BM25.
     *
     * @param query query string
     * @return best BM25 result, or null if no match
     */
    public BM25Result findBestIntent(String query) {
        List<BM25Result> results = rankNormalized(query, 1);
        return results.isEmpty() ? null : results.get(0);
    }

    /**
     * Find the best matching table for a query using BM25 over schema metadata.
     * Includes a bonus for direct table name match in the query.
     *
     * @param query        query string
     * @param schemaDocs   map of table name → BM25Document (pre-indexed schema)
     * @return best matching table name, or null if no match
     */
    public String findBestTable(String query, Map<String, BM25Document> schemaDocs) {
        List<String> queryTokens = tokenize(query);
        if (queryTokens.isEmpty() || schemaDocs.isEmpty()) {
            return schemaDocs.keySet().iterator().next();
        }

        String bestTable = null;
        double bestScore = -1;

        for (Map.Entry<String, BM25Document> e : schemaDocs.entrySet()) {
            double s = score(queryTokens, e.getValue());

            // Bonus: direct table name appears in query
            String tblLower = e.getKey().toLowerCase();
            if (query.toLowerCase().contains(tblLower)) {
                s += 3.0; // Strong bonus for direct table name match
            }

            if (s > bestScore) {
                bestScore = s;
                bestTable = e.getKey();
            }
        }

        return bestTable != null ? bestTable : schemaDocs.keySet().iterator().next();
    }

    // ═══════════════════════════════════════════════════════════
    // TEMPLATE INDEXING
    // ═══════════════════════════════════════════════════════════

    /**
     * Interface for SQL template providers.
     * Allows BM25 to index templates without depending on server-specific classes.
     */
    public interface TemplateProvider {
        /** Get all templates to index */
        List<TemplateEntry> getTemplates();

        /** A single template entry */
        record TemplateEntry(String id, String intent, String pattern, String description) {}
    }

    /**
     * Build BM25 index from SQL templates.
     * Each template's pattern + description + intent becomes a document.
     *
     * @param provider template provider
     */
    public void indexTemplates(TemplateProvider provider) {
        for (TemplateProvider.TemplateEntry t : provider.getTemplates()) {
            // Document = pattern (cleaned) + description + intent (rich source for matching)
            String docText = t.pattern().replace(".*", " ").replace("\\b", " ")
                .replace("|", " ") + " " + t.description() + " " + t.intent();
            addDocument("tpl_" + t.id(), t.intent(), docText, "template");
        }
        rebuildStats();
    }

    // ═══════════════════════════════════════════════════════════
    // SCHEMA INDEXING
    // ═══════════════════════════════════════════════════════════

    /**
     * Interface for schema metadata providers.
     * Allows BM25 to index schema without depending on server-specific classes.
     */
    public interface SchemaProvider {
        /** Get all schema entries to index */
        List<SchemaEntry> getSchemaEntries();

        /** A single schema entry */
        record SchemaEntry(
            String tableName,
            String tableComment,
            List<ColumnEntry> columns
        ) {}

        /** A column entry */
        record ColumnEntry(String name, String type, String comment) {}
    }

    /**
     * Build BM25 index from schema metadata (table names + COMMENT ON).
     * Each table becomes a document containing table name, comment, column names,
     * column comments, and type-hint synonyms.
     *
     * @param provider schema provider
     */
    public void indexSchema(SchemaProvider provider) {
        for (SchemaProvider.SchemaEntry se : provider.getSchemaEntries()) {
            StringBuilder sb = new StringBuilder();
            sb.append(se.tableName()).append(" ");
            sb.append(se.tableComment()).append(" ");

            for (SchemaProvider.ColumnEntry col : se.columns()) {
                sb.append(col.name()).append(" ");
                sb.append(col.comment()).append(" ");

                // Type-hint synonyms for cross-lingual matching
                String typeUpper = col.type().toUpperCase();
                if (typeUpper.contains("DECIMAL") || typeUpper.contains("INT")) {
                    sb.append("nombre montant valeur quantite numeric ");
                }
                if (typeUpper.contains("TIMESTAMP") || typeUpper.contains("DATE")) {
                    sb.append("date temps periode horodatage ");
                }
                if (typeUpper.contains("BOOLEAN")) {
                    sb.append("boolean indicateur drapeau flag ");
                }
                if (typeUpper.contains("VARCHAR") || typeUpper.contains("TEXT")) {
                    sb.append("texte nom libelle description ");
                }
            }

            addDocument("tbl_" + se.tableName().toLowerCase(), "", sb.toString(), "schema_table");
        }
        rebuildStats();
    }

    // ═══════════════════════════════════════════════════════════
    // ACCESSORS
    // ═══════════════════════════════════════════════════════════

    /** Get total number of indexed documents */
    public int getDocumentCount() { return totalDocs; }

    /** Get average document length */
    public double getAvgDocLength() { return avgDocLength; }

    /** Get the k1 parameter */
    public double getK1() { return k1; }

    /** Get the b parameter */
    public double getB() { return b; }

    /** Get document frequency for a term */
    public int getDocumentFrequency(String term) {
        return df.getOrDefault(term, 0);
    }

    /** Get a document by ID */
    public Optional<BM25Document> getDocument(String id) {
        return Optional.ofNullable(documents.get(id));
    }

    /** Get all documents */
    public Collection<BM25Document> getAllDocuments() {
        return Collections.unmodifiableCollection(documents.values());
    }

    /** Check if the index is empty */
    public boolean isEmpty() { return documents.isEmpty(); }

    @Override
    public String toString() {
        return String.format("BM25Ranker{docs=%d, avgLen=%.1f, k1=%.2f, b=%.2f}",
            totalDocs, avgDocLength, k1, b);
    }
}
