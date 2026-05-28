package com.llama4j;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * LLMAdapter — Pont entre Qwen35.java et SQL Assistant Enterprise
 * ============================================================
 * Wrapper qui encapsule le modele Qwen35 local pour :
 *   - Generer du SQL a partir de langage naturel
 *   - Ameliorer les requetes SQL existantes
 *   - Expliquer les requetes SQL en langage naturel
 *   - Suggirer des requetes pertinentes basees sur le schema
 *
 * Mode hybride :
 *   1. Ensemble Resolver -> intent + SQL draft (rapide, deterministe)
 *   2. LLM Qwen35 -> refine SQL + explain (precis, contextuel)
 *   3. Validate + execute
 */
public class LLMAdapter {

    private Qwen35 model;
    private Qwen35.State modelState;
    private final LLMConfig config;
    private volatile boolean loaded = false;
    private volatile boolean loading = false;
    private String loadError = null;

    // ─── Configuration ───────────────────────────────────────

    public static class LLMConfig {
        public String modelPath = "";
        public int contextLength = 2048;
        public float temperature = 0.3f;
        public float topP = 0.9f;
        public int maxTokens = 512;
        public boolean enabled = true;
        public String systemPrompt = "";
        public String modelLabel = "Qwen3.5";

        public LLMConfig() {}

        public static LLMConfig fromFile(String path) {
            LLMConfig cfg = new LLMConfig();
            try {
                String json = Files.readString(Paths.get(path));
                Map<String, String> m = parseSimpleJson(json);
                cfg.modelPath = m.getOrDefault("modelPath", "");
                cfg.contextLength = Integer.parseInt(m.getOrDefault("contextLength", "2048"));
                cfg.temperature = Float.parseFloat(m.getOrDefault("temperature", "0.3"));
                cfg.topP = Float.parseFloat(m.getOrDefault("topP", "0.9"));
                cfg.maxTokens = Integer.parseInt(m.getOrDefault("maxTokens", "512"));
                cfg.enabled = Boolean.parseBoolean(m.getOrDefault("enabled", "true"));
                cfg.systemPrompt = m.getOrDefault("systemPrompt", "");
                cfg.modelLabel = m.getOrDefault("modelLabel", "Qwen3.5");
            } catch (Exception e) {
                System.out.println("  [LLM] No config file or invalid: " + path + " (" + e.getMessage() + ")");
            }
            return cfg;
        }

        private static Map<String, String> parseSimpleJson(String json) {
            Map<String, String> map = new LinkedHashMap<>();
            json = json.trim();
            if (json.startsWith("{")) json = json.substring(1);
            if (json.endsWith("}")) json = json.substring(0, json.length() - 1);
            String[] parts = json.split(",");
            for (String part : parts) {
                String[] kv = part.split(":", 2);
                if (kv.length == 2) {
                    String key = kv[0].trim().replace("\"", "");
                    String val = kv[1].trim().replace("\"", "");
                    map.put(key, val);
                }
            }
            return map;
        }
    }

    // ─── Constructeur ────────────────────────────────────────

    public LLMAdapter(LLMConfig config) {
        this.config = config;
    }

    // ─── Chargement du modele (asynchrone) ───────────────────

    public synchronized void loadModel() {
        if (loaded || loading) return;
        if (config.modelPath == null || config.modelPath.isEmpty()) {
            loadError = "No model path configured";
            System.out.println("  [LLM] No model path configured - LLM mode disabled");
            config.enabled = false;
            return;
        }
        Path modelFile = Paths.get(config.modelPath);
        if (!Files.exists(modelFile)) {
            loadError = "Model file not found: " + config.modelPath;
            System.out.println("  [LLM] Model file not found: " + config.modelPath + " - LLM mode disabled");
            config.enabled = false;
            return;
        }

        loading = true;
        long start = System.currentTimeMillis();
        try {
            System.out.println("  [LLM] Loading model: " + config.modelPath + " (context=" + config.contextLength + ")...");
            model = ModelLoader.loadModel(modelFile, config.contextLength);
            modelState = model.createNewState();
            loaded = true;
            long elapsed = System.currentTimeMillis() - start;
            System.out.println("  [LLM] Model loaded successfully in " + elapsed + "ms (" + config.modelLabel + ")");
        } catch (Exception e) {
            loadError = "Failed to load model: " + e.getMessage();
            System.err.println("  [LLM] Error loading model: " + e.getMessage());
            config.enabled = false;
        } finally {
            loading = false;
        }
    }

    public void loadModelAsync() {
        CompletableFuture.runAsync(() -> {
            try {
                loadModel();
            } catch (Exception e) {
                System.err.println("  [LLM] Async load error: " + e.getMessage());
            }
        });
    }

    // ─── Generation de texte ─────────────────────────────────

    public String generate(String prompt) {
        return generate(prompt, config.maxTokens);
    }

    public String generate(String prompt, int maxTokens) {
        if (!isAvailable()) {
            return "[LLM not available" + (loadError != null ? ": " + loadError : "") + "]";
        }

        try {
            // Reset state for new conversation
            modelState = model.createNewState();

            // Use QwenChatFormat for proper chat encoding
            QwenChatFormat chatFormat = new QwenChatFormat(model.tokenizer);

            // Encode the prompt as a user message
            List<Integer> promptTokens = chatFormat.encodeMessage("user", prompt);
            // Add assistant header to prompt the model to respond
            promptTokens.addAll(chatFormat.encodeHeader("assistant", false));

            // Stop tokens - use EOS from tokenizer
            Set<Integer> stopTokens = new HashSet<>();
            stopTokens.add(model.tokenizer.getEosToken());

            // Sampler - CategoricalSampler with temperature
            Sampler sampler = new CategoricalSampler(
                java.util.random.RandomGeneratorFactory.of("L64X256MixRandom").create()
            );

            // Generate
            StringBuilder output = new StringBuilder();
            List<Integer> generated = Qwen35.generateTokens(
                model, modelState, 0, promptTokens,
                stopTokens, maxTokens, sampler, false, false,
                token -> {
                    String decoded = model.tokenizer.decode(List.of(token));
                    output.append(decoded);
                }
            );

            return output.toString().trim();
        } catch (Exception e) {
            return "[LLM generation error: " + e.getMessage() + "]";
        }
    }

    // ─── SQL-specific generation ─────────────────────────────

    /**
     * Genere du SQL a partir d'une question en langage naturel
     * en utilisant le schema de la base de donnees comme contexte.
     */
    public LLMSQLResult generateSQL(String question, String schemaContext, String dbName) {
        if (!isAvailable()) {
            return new LLMSQLResult("", "", 0.0, false, loadError);
        }

        String systemPart = config.systemPrompt.isEmpty()
            ? buildDefaultSystemPrompt(dbName)
            : config.systemPrompt;

        String prompt = systemPart + "\n\n"
            + "Database Schema:\n" + schemaContext + "\n\n"
            + "Question: " + question + "\n\n"
            + "SQL:";

        long start = System.currentTimeMillis();
        String raw = generate(prompt, Math.min(config.maxTokens, 256));
        long elapsed = System.currentTimeMillis() - start;

        // Extract SQL from response
        String sql = extractSQL(raw);
        String explanation = extractExplanation(raw);
        double confidence = estimateConfidence(raw, sql);

        return new LLMSQLResult(sql, explanation, confidence, true,
            "Generated in " + elapsed + "ms (" + config.modelLabel + ")");
    }

    /**
     * Ameliore une requete SQL existante avec le LLM.
     */
    public LLMSQLResult refineSQL(String existingSQL, String schemaContext, String dbName) {
        if (!isAvailable()) {
            return new LLMSQLResult(existingSQL, "", 0.0, false, loadError);
        }

        String prompt = "You are an SQL expert. Improve and optimize the following SQL query.\n"
            + "Database: " + dbName + "\n"
            + "Schema:\n" + schemaContext + "\n\n"
            + "Original SQL:\n" + existingSQL + "\n\n"
            + "Improved SQL:";

        String raw = generate(prompt, Math.min(config.maxTokens, 256));
        String sql = extractSQL(raw);
        String explanation = extractExplanation(raw);

        return new LLMSQLResult(sql, explanation, 0.85, true, "Refined by " + config.modelLabel);
    }

    /**
     * Explique une requete SQL en langage naturel.
     */
    public String explainSQL(String sql, String schemaContext) {
        if (!isAvailable()) {
            return "[LLM not available]";
        }

        String prompt = "Explain this SQL query in simple terms:\n\n"
            + "Schema:\n" + schemaContext + "\n\n"
            + "SQL: " + sql + "\n\n"
            + "Explanation:";

        return generate(prompt, Math.min(config.maxTokens, 200));
    }

    /**
     * Suggere des requetes pertinentes basees sur le schema.
     */
    public List<String> suggestQueries(String schemaContext, String dbName) {
        if (!isAvailable()) {
            return List.of();
        }

        String prompt = "Suggest 5 useful SQL queries for this database schema.\n"
            + "Database: " + dbName + "\n"
            + "Schema:\n" + schemaContext + "\n\n"
            + "Queries (one per line, SQL only):";

        String raw = generate(prompt, Math.min(config.maxTokens, 400));
        return Arrays.stream(raw.split("\n"))
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .filter(s -> s.toUpperCase().startsWith("SELECT")
                || s.toUpperCase().startsWith("INSERT")
                || s.toUpperCase().startsWith("UPDATE")
                || s.toUpperCase().startsWith("DELETE")
                || s.toUpperCase().startsWith("WITH"))
            .limit(5)
            .toList();
    }

    // ─── Helpers ─────────────────────────────────────────────

    private String buildDefaultSystemPrompt(String dbName) {
        return "You are an expert SQL assistant for the \"" + dbName + "\" database. "
            + "Generate only valid SQL queries. "
            + "Do not include explanations in the SQL output. "
            + "Use only the tables and columns from the provided schema. "
            + "Always use proper SQL syntax compatible with PostgreSQL/H2.";
    }

    private String extractSQL(String raw) {
        // Try to extract SQL from markdown code blocks
        String sql = raw;

        // Remove markdown code blocks
        if (sql.contains("```sql")) {
            int start = sql.indexOf("```sql") + 6;
            int end = sql.indexOf("```", start);
            if (end > start) {
                sql = sql.substring(start, end).trim();
            }
        } else if (sql.contains("```")) {
            int start = sql.indexOf("```") + 3;
            int end = sql.indexOf("```", start);
            if (end > start) {
                sql = sql.substring(start, end).trim();
            }
        }

        // Clean up
        sql = sql.replaceAll("(?i)^SQL:?\\s*", "").trim();

        // Remove trailing explanation after semicolon
        int semiPos = sql.indexOf(';');
        if (semiPos > 0 && semiPos < sql.length() - 1) {
            String afterSemi = sql.substring(semiPos + 1).trim();
            if (afterSemi.length() > 20) {
                sql = sql.substring(0, semiPos + 1);
            }
        }

        return sql.trim();
    }

    private String extractExplanation(String raw) {
        // Extract text after SQL statement
        int semiPos = raw.indexOf(';');
        if (semiPos > 0 && semiPos < raw.length() - 1) {
            String after = raw.substring(semiPos + 1).trim();
            if (after.length() > 10 && !after.toUpperCase().startsWith("SELECT")
                && !after.toUpperCase().startsWith("FROM")
                && !after.toUpperCase().startsWith("WHERE")) {
                return after;
            }
        }
        return "";
    }

    private double estimateConfidence(String raw, String sql) {
        if (sql.isEmpty()) return 0.1;
        double conf = 0.5;
        String upper = sql.toUpperCase();
        if (upper.startsWith("SELECT")) conf += 0.15;
        if (upper.contains("FROM")) conf += 0.1;
        if (upper.contains("WHERE")) conf += 0.05;
        if (upper.contains("JOIN")) conf += 0.05;
        if (upper.contains("GROUP BY")) conf += 0.05;
        if (sql.endsWith(";")) conf += 0.05;
        if (raw.contains("I cannot") || raw.contains("error") || raw.contains("invalid")) conf -= 0.2;
        return Math.max(0.1, Math.min(0.99, conf));
    }

    // ─── Status ──────────────────────────────────────────────

    public boolean isAvailable() {
        return loaded && config.enabled && model != null;
    }

    public boolean isLoading() {
        return loading;
    }

    public Map<String, Object> getStatus() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("available", isAvailable());
        status.put("loading", loading);
        status.put("enabled", config.enabled);
        status.put("modelLabel", config.modelLabel);
        status.put("modelPath", config.modelPath);
        status.put("contextLength", config.contextLength);
        status.put("temperature", (double) config.temperature);
        status.put("maxTokens", config.maxTokens);
        if (loadError != null) status.put("error", loadError);
        return status;
    }

    // ─── Result record ───────────────────────────────────────

    public record LLMSQLResult(String sql, String explanation, double confidence,
                        boolean llmUsed, String meta) {}
}
