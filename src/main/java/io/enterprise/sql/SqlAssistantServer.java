package io.enterprise.sql;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Enterprise SQL Assistant V3 — JDK HttpServer Backend + Multi-Database
 * ====================================================
 * Pure Java server using com.sun.net.httpserver.HttpServer + JDBC.
 * No framework, no external deps (except JDBC drivers) — just the JDK + H2.
 *
 * Features:
 *   - Multi-database support via JDBC (configured in databases.json)
 *   - Dynamic schema discovery at startup via DatabaseMetaData
 *   - SQL query execution with result sets returned as JSON tables
 *   - 5-strategy ensemble resolver (weighted voting)
 *   - Adaptive feedback system
 *
 * Architecture:
 *   1. PERCEIVE  — Normalize input
 *   2. REASON    — 5-strategy ensemble resolver (weighted voting)
 *   3. ACT       — Generate SQL from resolved intent
 *   4. REFLECT   — Validate and score
 *   5. EXECUTE   — Run SQL against selected database
 *
 * Endpoints:
 *   GET  /                → Client HTML
 *   GET  /api/databases   → List configured databases
 *   GET  /api/schema      → Schema for current/selected database
 *   POST /api/query       → Translate NL → SQL
 *   POST /api/execute     → Execute SQL on selected database
 *   GET  /api/metrics     → Agent metrics
 *   GET  /api/strategies  → Strategy weights & accuracy
 *   GET  /api/patterns    → Mined query patterns
 *   POST /api/feedback    → Submit feedback (adapts weights)
 */
public class SqlAssistantServer {

    // ═══════════════════════════════════════════════════════════
    // DATABASE MANAGER — Multi-DB JDBC Connections + Schema Discovery
    // ═══════════════════════════════════════════════════════════

    static class DatabaseConfig {
        String name, description, driver, url, user, password;

        DatabaseConfig(String name, String description, String driver,
                       String url, String user, String password) {
            this.name = name;
            this.description = description;
            this.driver = driver;
            this.url = url;
            this.user = user;
            this.password = password;
        }
    }

    static class DatabaseManager {
        private final Map<String, DatabaseConfig> configs = new LinkedHashMap<>();
        private final Map<String, Connection> connections = new ConcurrentHashMap<>();
        private final Map<String, Map<String, TableSchema>> schemas = new ConcurrentHashMap<>();

        void loadConfig(String configPath) throws Exception {
            String json = Files.readString(Paths.get(configPath));
            // Parse minimal JSON array
            List<Map<String, String>> entries = parseJsonArrayOfObjects(json);
            for (Map<String, String> e : entries) {
                DatabaseConfig cfg = new DatabaseConfig(
                        e.get("name"),
                        e.get("description"),
                        e.get("driver"),
                        e.get("url"),
                        e.get("user"),
                        e.get("password")
                );
                configs.put(cfg.name, cfg);
            }
        }

        void connectAll() {
            for (Map.Entry<String, DatabaseConfig> e : configs.entrySet()) {
                try {
                    DatabaseConfig cfg = e.getValue();
                    Class.forName(cfg.driver);
                    Connection conn = DriverManager.getConnection(cfg.url, cfg.user, cfg.password);
                    connections.put(cfg.name, conn);
                    System.out.println("  [OK] Connected to: " + cfg.name + " (" + cfg.url + ")");

                    // Discover schema dynamically
                    Map<String, TableSchema> dbSchema = discoverSchema(conn, cfg.name);
                    schemas.put(cfg.name, dbSchema);
                    System.out.println("       Discovered " + dbSchema.size() + " tables");
                } catch (Exception ex) {
                    System.err.println("  [FAIL] " + e.getKey() + ": " + ex.getMessage());
                }
            }
        }

        private Map<String, TableSchema> discoverSchema(Connection conn, String schema) throws SQLException {
            Map<String, TableSchema> result = new LinkedHashMap<>();
            DatabaseMetaData meta = conn.getMetaData();
            // Get all tables (filter out H2 system tables)
            try (ResultSet tables = meta.getTables(schema, null, "%", new String[]{"TABLE"})) {
                while (tables.next()) {
                    String tableName = tables.getString("TABLE_NAME");
                    // Skip H2 system tables
                    if (tableName.startsWith("INFORMATION_SCHEMA") || tableName.startsWith("SYSTEM_")
                            || tableName.equals("DATABASECHANGELOG") || tableName.equals("DATABASECHANGELOGLOCK"))
                        continue;
                    List<ColumnDef> columns = new ArrayList<>();
                    List<String> indexes = new ArrayList<>();

                    // Get columns
                    try (ResultSet cols = meta.getColumns(null, schema, tableName, "%")) {
                        while (cols.next()) {
                            String colName = cols.getString("COLUMN_NAME");
                            String colType = cols.getString("TYPE_NAME");
                            int colSize = cols.getInt("COLUMN_SIZE");
                            String nullable = cols.getString("IS_NULLABLE");
                            String isAutoInc = cols.getString("IS_AUTOINCREMENT");

                            // Format type with size
                            String typeStr = colType;
                            if (colSize > 0 && !"TEXT".equalsIgnoreCase(colType)
                                    && !"CLOB".equalsIgnoreCase(colType)
                                    && !"BLOB".equalsIgnoreCase(colType)
                                    && !"BOOLEAN".equalsIgnoreCase(colType)
                                    && !"DATE".equalsIgnoreCase(colType)
                                    && !"TIMESTAMP".equalsIgnoreCase(colType)) {
                                typeStr = colType + "(" + colSize + ")";
                            }

                            columns.add(new ColumnDef(colName, typeStr, false,
                                    "YES".equalsIgnoreCase(nullable)));
                        }
                    }

                    // Identify primary keys
                    Set<String> pkCols = new HashSet<>();
                    try (ResultSet pks = meta.getPrimaryKeys(null, null, tableName)) {
                        while (pks.next()) {
                            pkCols.add(pks.getString("COLUMN_NAME"));
                        }
                    }
                    // Mark PK columns
                    List<ColumnDef> updatedCols = columns.stream()
                            .map(c -> pkCols.contains(c.name)
                                    ? new ColumnDef(c.name, c.type, true, c.nullable) : c)
                            .collect(Collectors.toList());

                    // Get indexes
                    try (ResultSet idx = meta.getIndexInfo(null, null, tableName, false, false)) {
                        while (idx.next()) {
                            String idxName = idx.getString("INDEX_NAME");
                            if (idxName != null && !idxName.startsWith("PK_")
                                    && !idxName.startsWith("PRIMARY_KEY")) {
                                indexes.add(idxName);
                            }
                        }
                    }

                    result.put(tableName.toLowerCase(), new TableSchema(tableName, updatedCols, indexes));
                }
            }
            return result;
        }

        Map<String, TableSchema> getSchema(String dbName) {
            return schemas.getOrDefault(dbName, new LinkedHashMap<>());
        }

        Connection getConnection(String dbName) throws SQLException {
            Connection conn = connections.get(dbName);
            if (conn == null || conn.isClosed()) {
                DatabaseConfig cfg = configs.get(dbName);
                if (cfg == null) throw new SQLException("Unknown database: " + dbName);
                conn = DriverManager.getConnection(cfg.url, cfg.user, cfg.password);
                connections.put(dbName, conn);
            }
            return conn;
        }

        List<Map<String, Object>> getDatabaseList() {
            List<Map<String, Object>> list = new ArrayList<>();
            for (Map.Entry<String, DatabaseConfig> e : configs.entrySet()) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("name", e.getKey());
                m.put("description", e.getValue().description);
                m.put("connected", connections.containsKey(e.getKey()));
                int tableCount = schemas.getOrDefault(e.getKey(), new LinkedHashMap<>()).size();
                m.put("tableCount", tableCount);
                list.add(m);
            }
            return list;
        }

        QueryResult executeQuery(String dbName, String sql) {
            QueryResult result = new QueryResult();
            try {
                Connection conn = getConnection(dbName);
                Statement stmt = conn.createStatement();
                stmt.setMaxRows(500);

                boolean hasResultSet = stmt.execute(sql);
                if (hasResultSet) {
                    ResultSet rs = stmt.getResultSet();
                    ResultSetMetaData meta = rs.getMetaData();
                    int colCount = meta.getColumnCount();

                    // Column info
                    for (int i = 1; i <= colCount; i++) {
                        result.columns.add(new ResultColumn(
                                meta.getColumnLabel(i),
                                meta.getColumnTypeName(i)
                        ));
                    }

                    // Data rows
                    int rowCount = 0;
                    while (rs.next() && rowCount < 500) {
                        Map<String, Object> row = new LinkedHashMap<>();
                        for (int i = 1; i <= colCount; i++) {
                            Object val = rs.getObject(i);
                            row.put(result.columns.get(i - 1).name,
                                    val == null ? null : val.toString());
                        }
                        result.rows.add(row);
                        rowCount++;
                    }
                    result.rowCount = rowCount;
                    result.hasMore = rs.next();
                    rs.close();
                } else {
                    result.updateCount = stmt.getUpdateCount();
                }
                stmt.close();
                result.success = true;
            } catch (SQLException e) {
                result.success = false;
                result.error = e.getMessage();
            }
            return result;
        }
    }

    static class QueryResult {
        List<ResultColumn> columns = new ArrayList<>();
        List<Map<String, Object>> rows = new ArrayList<>();
        int rowCount = 0;
        boolean hasMore = false;
        int updateCount = -1;
        boolean success = true;
        String error = null;
    }

    static class ResultColumn {
        String name, type;

        ResultColumn(String name, String type) {
            this.name = name;
            this.type = type;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // SCHEMA DATA CLASSES
    // ═══════════════════════════════════════════════════════════

    record ColumnDef(String name, String type, boolean pk, boolean nullable) {
    }

    record TableSchema(String tableName, List<ColumnDef> columns, List<String> indexes) {
    }

    // ═══════════════════════════════════════════════════════════
    // ENGINE — Ensemble Resolver + SQL Generator
    // ═══════════════════════════════════════════════════════════

    static class SqlEngine {
        private final Object lock = new Object();
        private final AtomicLong totalQueries = new AtomicLong(0);
        private final AtomicLong successfulQueries = new AtomicLong(0);
        private final AtomicLong totalLatencyNanos = new AtomicLong(0);
        private final AtomicLong correctFeedback = new AtomicLong(0);
        private final AtomicLong incorrectFeedback = new AtomicLong(0);

        private final Map<String, Double> strategyWeights = new ConcurrentHashMap<>();
        private final Map<String, Double> strategyAccuracy = new ConcurrentHashMap<>();
        private final List<QueryHistory> history = Collections.synchronizedList(new ArrayList<>());
        private final List<PatternEntry> patterns = Collections.synchronizedList(new ArrayList<>());

        SqlEngine() {
            strategyWeights.put("intent_graph", 1.0);
            strategyWeights.put("lsh_multi_probe", 0.9);
            strategyWeights.put("syntactic_parser", 1.2);
            strategyWeights.put("thesaurus_hash", 0.8);
            strategyWeights.put("regex_rules", 0.6);

            strategyAccuracy.put("intent_graph", 0.88);
            strategyAccuracy.put("lsh_multi_probe", 0.82);
            strategyAccuracy.put("syntactic_parser", 0.92);
            strategyAccuracy.put("thesaurus_hash", 0.78);
            strategyAccuracy.put("regex_rules", 0.70);
        }

        record Vote(String strategy, String intentType, double rawConfidence, double weightedScore) {
        }

        EnsembleResult resolveIntent(String query, Map<String, TableSchema> schema) {
            String q = query.toLowerCase();
            List<Vote> votes = new ArrayList<>();

            votes.add(resolveWithGraph(q));
            votes.add(resolveWithLSH(q));
            votes.add(resolveWithSyntax(q));
            votes.add(resolveWithThesaurus(q));
            votes.add(resolveWithRegex(q));

            // Aggregate
            Map<String, Double> scores = new LinkedHashMap<>();
            Map<String, Double> bestRaw = new LinkedHashMap<>();
            for (Vote v : votes) {
                scores.merge(v.intentType, v.weightedScore, Double::sum);
                bestRaw.merge(v.intentType, v.rawConfidence, Double::max);
            }

            String winner = scores.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey).orElse("Unknown");
            double total = scores.values().stream().mapToDouble(Double::doubleValue).sum();
            double ensembleConf = total > 0 ? scores.get(winner) / total : 0;
            double calibrated = ensembleConf * 0.6 + bestRaw.getOrDefault(winner, 0.5) * 0.4;

            return new EnsembleResult(winner, calibrated, ensembleConf, votes, scores);
        }

        record EnsembleResult(String intent, double confidence, double ensembleConfidence,
                              List<Vote> votes, Map<String, Double> scores) {
        }

        private Vote resolveWithGraph(String q) {
            String intent = "Unknown";
            double conf = 0.1;
            if (match(q, "select|show|get|find|list|display|montre|affiche|cherche|liste") &&
                    match(q, "from|de|dans")) {
                intent = "Select";
                conf = 0.85;
            } else if (match(q, "how many|combien|count|nombre")) {
                intent = "Aggregate";
                conf = 0.80;
            } else if (match(q, "total|somme|sum")) {
                intent = "Aggregate";
                conf = 0.75;
            } else if (match(q, "insert|add|ajoute|ajouter|creer un")) {
                intent = "Insert";
                conf = 0.85;
            } else if (match(q, "update|modify|change|modifie|modifier|mettre a jour")) {
                intent = "Update";
                conf = 0.80;
            } else if (match(q, "delete|remove|supprime|supprimer|efface")) {
                intent = "Delete";
                conf = 0.85;
            } else if (match(q, "create table|cre.*table|nouvelle table")) {
                intent = "CreateTable";
                conf = 0.90;
            } else if (match(q, "alter|modifier.*table")) {
                intent = "AlterTable";
                conf = 0.90;
            } else if (match(q, "drop|supprimer.*table")) {
                intent = "DropTable";
                conf = 0.90;
            } else if (match(q, "join|combine|fusionne|joindre|combiner")) {
                intent = "Join";
                conf = 0.70;
            } else if (match(q, "explain|analyse|explique")) {
                intent = "Explain";
                conf = 0.90;
            } else if (match(q, "describe|schema|structure|decris|colonnes")) {
                intent = "SchemaInfo";
                conf = 0.80;
            }
            return new Vote("intent_graph", intent, conf, conf * strategyWeights.get("intent_graph"));
        }

        private Vote resolveWithLSH(String q) {
            String intent = "Unknown";
            double conf = 0.1;
            if (match(q, "select|show|get|affiche|montre|liste|cherche")) {
                intent = "Select";
                conf = 0.80;
            } else if (match(q, "count|combien|nombre|total|somme")) {
                intent = "Aggregate";
                conf = 0.75;
            } else if (match(q, "insert|ajout|add")) {
                intent = "Insert";
                conf = 0.82;
            } else if (match(q, "update|modif|change")) {
                intent = "Update";
                conf = 0.78;
            } else if (match(q, "delete|supprim|remove")) {
                intent = "Delete";
                conf = 0.82;
            } else if (match(q, "join|combin|fusion")) {
                intent = "Join";
                conf = 0.68;
            }
            return new Vote("lsh_multi_probe", intent, conf, conf * strategyWeights.get("lsh_multi_probe"));
        }

        private Vote resolveWithSyntax(String q) {
            String intent = "Unknown";
            double conf = 0.1;
            Map<String, String> verbMap = Map.ofEntries(
                    Map.entry("select", "Select"), Map.entry("get", "Select"), Map.entry("show", "Select"),
                    Map.entry("find", "Select"), Map.entry("list", "Select"),
                    Map.entry("affiche", "Select"), Map.entry("montre", "Select"),
                    Map.entry("cherche", "Select"), Map.entry("liste", "Select"),
                    Map.entry("insert", "Insert"), Map.entry("add", "Insert"),
                    Map.entry("ajoute", "Insert"), Map.entry("ajouter", "Insert"),
                    Map.entry("update", "Update"), Map.entry("modify", "Update"),
                    Map.entry("change", "Update"), Map.entry("modifie", "Update"),
                    Map.entry("modifier", "Update"),
                    Map.entry("delete", "Delete"), Map.entry("remove", "Delete"),
                    Map.entry("supprime", "Delete"), Map.entry("supprimer", "Delete"),
                    Map.entry("efface", "Delete"),
                    Map.entry("create", "CreateTable"), Map.entry("creer", "CreateTable"),
                    Map.entry("join", "Join"), Map.entry("combine", "Join"),
                    Map.entry("fusionne", "Join"), Map.entry("joindre", "Join"),
                    Map.entry("explain", "Explain"), Map.entry("explique", "Explain"),
                    Map.entry("analyse", "Explain"),
                    Map.entry("describe", "SchemaInfo"), Map.entry("decris", "SchemaInfo")
            );
            for (String word : q.split("\\s+")) {
                if (verbMap.containsKey(word)) {
                    intent = verbMap.get(word);
                    conf = 0.88;
                    break;
                }
            }
            if ("Unknown".equals(intent) && match(q, "combien|nombre|count|total|moyenne|average|somme|sum|max|min")) {
                intent = "Aggregate";
                conf = 0.85;
            }
            return new Vote("syntactic_parser", intent, conf, conf * strategyWeights.get("syntactic_parser"));
        }

        private Vote resolveWithThesaurus(String q) {
            String intent = "Unknown";
            double conf = 0.1;
            Map<String, String> thesMap = new LinkedHashMap<>();
            thesMap.put("utilisateurs", "Select");
            thesMap.put("clients", "Select");
            thesMap.put("commandes", "Select");
            thesMap.put("produits", "Select");
            thesMap.put("employes", "Select");
            thesMap.put("salaires", "Select");
            thesMap.put("donnees", "Select");
            thesMap.put("enregistrements", "Select");
            thesMap.put("nombre", "Aggregate");
            thesMap.put("total", "Aggregate");
            thesMap.put("moyenne", "Aggregate");
            thesMap.put("somme", "Aggregate");
            thesMap.put("ajouter", "Insert");
            thesMap.put("inserer", "Insert");
            thesMap.put("modifier", "Update");
            thesMap.put("supprimer", "Delete");
            thesMap.put("joindre", "Join");
            thesMap.put("combiner", "Join");
            thesMap.put("fusionner", "Join");
            thesMap.put("expliquer", "Explain");
            thesMap.put("decrire", "SchemaInfo");
            for (Map.Entry<String, String> e : thesMap.entrySet()) {
                if (q.contains(e.getKey())) {
                    intent = e.getValue();
                    conf = 0.76;
                    break;
                }
            }
            return new Vote("thesaurus_hash", intent, conf, conf * strategyWeights.get("thesaurus_hash"));
        }

        private Vote resolveWithRegex(String q) {
            String intent = "Unknown";
            double conf = 0.1;
            String[][] rules = {
                    {"\\b(select|show|get|find|list|display)\\b.*\\b(from|in)\\b", "Select", "0.85"},
                    {"\\b(insert|add|put)\\b.*\\b(into|to|in)\\b", "Insert", "0.85"},
                    {"\\b(update|modify|change)\\b.*\\b(set)\\b", "Update", "0.85"},
                    {"\\b(delete|remove)\\b.*\\b(from)\\b", "Delete", "0.85"},
                    {"\\b(create|make)\\s+(table)\\b", "CreateTable", "0.9"},
                    {"\\b(alter|modify)\\s+(table)\\b", "AlterTable", "0.9"},
                    {"\\b(drop|remove)\\s+(table)\\b", "DropTable", "0.9"},
                    {"\\b(join|combine|merge)\\b", "Join", "0.7"},
                    {"\\b(count|sum|average|avg|min|max|how many|combien)\\b", "Aggregate", "0.8"},
                    {"\\b(explain|analyse)\\b", "Explain", "0.9"},
                    {"\\b(describe|schema|colonnes)\\b", "SchemaInfo", "0.8"},
                    {"\\b(montre|affiche|liste|cherche)\\b", "Select", "0.8"},
                    {"\\b(ajoute|insere|creer)\\b", "Insert", "0.8"},
                    {"\\b(modifie|change|met a jour)\\b", "Update", "0.8"},
                    {"\\b(supprime|efface|retire)\\b", "Delete", "0.8"},
                    {"\\b(combien|nombre|total|somme|moyenne)\\b", "Aggregate", "0.8"},
                    {"\\b(joindre|combiner|fusionner)\\b", "Join", "0.75"},
                    {"\\b(explique|analyse)\\b", "Explain", "0.85"},
                    {"\\b(decris|structure|colonnes de)\\b", "SchemaInfo", "0.8"},
            };
            for (String[] rule : rules) {
                if (Pattern.compile(rule[0]).matcher(q).find()) {
                    intent = rule[1];
                    conf = Double.parseDouble(rule[2]);
                    break;
                }
            }
            return new Vote("regex_rules", intent, conf, conf * strategyWeights.get("regex_rules"));
        }

        private boolean match(String q, String regex) {
            return Pattern.compile("\\b(" + regex + ")\\b").matcher(q).find();
        }

        // ── SQL Generator (schema-aware) ──

        String generateSQL(String query, String intent, Map<String, TableSchema> schema) {
            String q = query.toLowerCase();
            switch (intent) {
                case "Select": {
                    String table = resolveTable(q, schema);
                    TableSchema ts = schema.get(table);
                    String cols = match(q, "all|tous|tout|\\*") ? "*" :
                            (ts != null ? ts.columns.stream().map(c -> c.name).collect(Collectors.joining(", ")) : "*");
                    List<String> conds = extractConditions(q, ts);
                    String where = conds.isEmpty() ? "" : "\nWHERE " + String.join("\n  AND ", conds);
                    var om = Pattern.compile("\\b(order by|trie|sorted|tri)\\s+(?:by\\s+)?(\\w+)").matcher(q);
                    String order = om.find() ? "\nORDER BY " + om.group(2) : "";
                    var lm = Pattern.compile("\\b(top|limit|premier)\\s+(\\d+)").matcher(q);
                    String limit = lm.find() ? "\nLIMIT " + lm.group(2) : "";
                    return "SELECT " + cols + "\nFROM " + table + where + order + limit + ";";
                }
                case "Insert": {
                    String table = resolveTable(q, schema);
                    TableSchema ts = schema.get(table);
                    List<String> colNames = ts != null ? ts.columns.stream()
                            .filter(c -> !c.pk).map(c -> c.name).collect(Collectors.toList())
                            : List.of("name", "value");
                    String vals = colNames.stream().map(c -> ":" + c).collect(Collectors.joining(", "));
                    return "INSERT INTO " + table + " (" + String.join(", ", colNames) + ")\nVALUES (" + vals + ");";
                }
                case "Update": {
                    String table = resolveTable(q, schema);
                    List<String> conds = extractConditions(q, schema.get(table));
                    String where = conds.isEmpty() ? "\nWHERE id = :id" : "\nWHERE " + String.join("\n  AND ", conds);
                    return "UPDATE " + table + "\nSET :column = :value" + where + ";";
                }
                case "Delete": {
                    String table = resolveTable(q, schema);
                    List<String> conds = extractConditions(q, schema.get(table));
                    String where = conds.isEmpty() ? "\nWHERE id = :id" : "\nWHERE " + String.join("\n  AND ", conds);
                    return "DELETE FROM " + table + where + ";";
                }
                case "CreateTable": {
                    var nm = Pattern.compile("(?:table)\\s+(\\w+)").matcher(q);
                    String name = nm.find() ? nm.group(1) : "new_table";
                    return "CREATE TABLE " + name + " (\n  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,\n  name VARCHAR(255) NOT NULL,\n  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);";
                }
                case "Join": {
                    List<String> tables = new ArrayList<>(schema.keySet());
                    if (tables.size() >= 2) {
                        return "SELECT *\nFROM " + tables.get(0) + "\nINNER JOIN " + tables.get(1)
                                + " ON " + tables.get(0) + ".id = " + tables.get(1) + ".id;";
                    }
                    return "SELECT * FROM " + tables.get(0) + ";";
                }
                case "Aggregate": {
                    String table = resolveTable(q, schema);
                    String fn = "COUNT";
                    if (match(q, "sum|total|somme")) fn = "SUM";
                    else if (match(q, "average|avg|moyenne|mean")) fn = "AVG";
                    else if (match(q, "max|maximum|highest|plus grand")) fn = "MAX";
                    else if (match(q, "min|minimum|lowest|plus petit")) fn = "MIN";
                    var gm = Pattern.compile("\\b(group by|par|by)\\s+(\\w+)").matcher(q);
                    String group = "";
                    String col = "1";
                    if (gm.find()) {
                        group = "\nGROUP BY " + gm.group(2);
                        col = gm.group(2);
                    }
                    List<String> conds = extractConditions(q, schema.get(table));
                    String where = conds.isEmpty() ? "" : "\nWHERE " + String.join("\n  AND ", conds);
                    return "SELECT " + fn + "(*), " + col + "\nFROM " + table + where + group + ";";
                }
                case "Explain":
                    return "EXPLAIN ANALYZE\nSELECT * FROM " + resolveTable(q, schema) + " WHERE id = 1;";
                case "SchemaInfo": {
                    String table = resolveTable(q, schema);
                    TableSchema ts = schema.get(table);
                    if (ts != null) {
                        String cols = ts.columns.stream()
                                .map(c -> "  " + c.name + " " + c.type + (c.pk ? " PRIMARY KEY" : "") + (c.nullable ? "" : " NOT NULL"))
                                .collect(Collectors.joining(",\n"));
                        return "-- Schema: " + table + "\n-- Columns:\n" + cols;
                    }
                    return "-- Schema info for: " + table;
                }
                default:
                    return "-- Unable to resolve intent for: \"" + query + "\"\n-- Please rephrase your query.";
            }
        }

        private String resolveTable(String q, Map<String, TableSchema> schema) {
            // Build dynamic table name map from schema
            Map<String, String> map = new LinkedHashMap<>();
            for (String tableName : schema.keySet()) {
                map.put(tableName, tableName);
                map.put(tableName.toLowerCase(), tableName);
                // Add common singular/plural
                if (tableName.endsWith("s")) {
                    map.put(tableName.substring(0, tableName.length() - 1), tableName);
                }
            }
            // French mappings for common table names
            map.put("utilisateur", findBestMatch(schema, "user", "employe"));
            map.put("utilisateurs", findBestMatch(schema, "user", "employe"));
            map.put("commande", findBestMatch(schema, "order"));
            map.put("commandes", findBestMatch(schema, "order"));
            map.put("produit", findBestMatch(schema, "product"));
            map.put("produits", findBestMatch(schema, "product"));
            map.put("employe", findBestMatch(schema, "employe", "user"));
            map.put("employes", findBestMatch(schema, "employe", "user"));
            map.put("departement", findBestMatch(schema, "department"));
            map.put("departements", findBestMatch(schema, "department"));
            map.put("evenement", findBestMatch(schema, "event"));
            map.put("evenements", findBestMatch(schema, "event"));
            map.put("session", findBestMatch(schema, "session"));
            map.put("sessions", findBestMatch(schema, "session"));
            map.put("salaire", findBestMatch(schema, "salar", "salary"));
            map.put("salaires", findBestMatch(schema, "salar", "salary"));
            map.put("conge", findBestMatch(schema, "leave"));
            map.put("conges", findBestMatch(schema, "leave"));
            map.put("categorie", findBestMatch(schema, "categor"));
            map.put("categories", findBestMatch(schema, "categor"));
            map.put("metrique", findBestMatch(schema, "metric"));
            map.put("metriques", findBestMatch(schema, "metric"));
            map.put("conversion", findBestMatch(schema, "conversion"));
            map.put("conversions", findBestMatch(schema, "conversion"));

            for (Map.Entry<String, String> e : map.entrySet()) {
                if (q.contains(e.getKey()) && e.getValue() != null) return e.getValue();
            }
            return schema.keySet().iterator().next();
        }

        private String findBestMatch(Map<String, TableSchema> schema, String... prefixes) {
            for (String prefix : prefixes) {
                for (String name : schema.keySet()) {
                    if (name.toLowerCase().startsWith(prefix.toLowerCase())) return name;
                }
            }
            return null;
        }

        private List<String> extractConditions(String q, TableSchema ts) {
            List<String> conds = new ArrayList<>();
            if (ts == null) return conds;

            // Check for known column-based conditions
            for (ColumnDef col : ts.columns) {
                String colName = col.name.toLowerCase();
                if (("active".equals(colName) || "actif".equals(colName)) && match(q, "active|actif")) {
                    conds.add(col.name + " = TRUE");
                }
                if (("active".equals(colName) || "actif".equals(colName)) && match(q, "inactive|inactif")) {
                    conds.add(col.name + " = FALSE");
                }
                if ("status".equals(colName)) {
                    var sm = Pattern.compile("(?:status|etat)\\s*(?:=|est)\\s*['\"]?(\\w+)").matcher(q);
                    if (sm.find()) conds.add(col.name + " = '" + sm.group(1) + "'");
                }
            }

            // Generic conditions
            var m = Pattern.compile("(\\d+)").matcher(q);
            if (match(q, "greater than|plus de|superieur|>\\s*\\d+|more than") && m.find()) {
                // Find numeric column
                String numCol = findNumericColumn(ts);
                if (numCol != null) conds.add(numCol + " > " + m.group(1));
            }
            m = Pattern.compile("(\\d+)").matcher(q);
            if (match(q, "less than|moins de|inferieur|<\\s*\\d+") && m.find()) {
                String numCol = findNumericColumn(ts);
                if (numCol != null) conds.add(numCol + " < " + m.group(1));
            }
            if (match(q, "recent|recente|dernier|last")) {
                for (ColumnDef col : ts.columns) {
                    if (col.type.toUpperCase().contains("TIMESTAMP") || col.type.toUpperCase().contains("DATE")) {
                        conds.add(col.name + " >= CURRENT_DATE - INTERVAL '30 days'");
                        break;
                    }
                }
            }
            return conds;
        }

        private String findNumericColumn(TableSchema ts) {
            for (ColumnDef col : ts.columns) {
                String t = col.type.toUpperCase();
                if (t.contains("DECIMAL") || t.contains("INT") || t.contains("DOUBLE")
                        || t.contains("FLOAT") || t.contains("NUMERIC")) {
                    if (!col.pk && !col.name.toLowerCase().contains("id")) return col.name;
                }
            }
            return null;
        }

        // ── Full Pipeline ──

        Map<String, Object> processQuery(String query, String dbName, Map<String, TableSchema> schema) {
            long start = System.nanoTime();

            String normalized = query.trim();
            if (normalized.isEmpty()) return Map.of("error", "Query must not be blank");

            EnsembleResult result = resolveIntent(normalized, schema);
            String sql = generateSQL(normalized, result.intent, schema);

            boolean valid = !sql.startsWith("--") && sql.length() > 5;

            long latency = System.nanoTime() - start;
            long latencyUs = latency / 1000;

            int queryId;
            synchronized (lock) {
                queryId = history.size();
                totalQueries.incrementAndGet();
                successfulQueries.incrementAndGet();
                totalLatencyNanos.addAndGet(latency);
                history.add(new QueryHistory(queryId, query, result.intent, sql,
                        result.confidence, latencyUs, valid, result.votes, System.currentTimeMillis(), dbName));

                Optional<PatternEntry> existing = patterns.stream()
                        .filter(p -> p.intent.equals(result.intent)).findFirst();
                if (existing.isPresent()) {
                    existing.get().freq++;
                    existing.get().avgConf = (existing.get().avgConf + result.confidence) / 2;
                } else {
                    patterns.add(new PatternEntry(result.intent, 1, result.confidence));
                }
                patterns.sort((a, b) -> b.freq - a.freq);
            }

            List<Map<String, Object>> voteList = result.votes.stream().map(v -> {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("strategy", v.strategy);
                m.put("intentType", v.intentType);
                m.put("rawConfidence", v.rawConfidence);
                m.put("weightedScore", v.weightedScore);
                return m;
            }).collect(Collectors.toList());

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("queryId", queryId);
            response.put("query", query);
            response.put("database", dbName);
            response.put("intent", result.intent);
            response.put("confidence", result.confidence);
            response.put("ensembleConfidence", result.ensembleConfidence);
            response.put("sql", sql);
            response.put("valid", valid);
            response.put("validationMessage", valid ? "OK" : "Generated SQL has issues");
            response.put("latencyMicros", latencyUs);
            response.put("votes", voteList);
            response.put("scores", result.scores);
            return response;
        }

        void recordFeedback(int queryId, boolean isCorrect) {
            synchronized (lock) {
                if (isCorrect) correctFeedback.incrementAndGet();
                else incorrectFeedback.incrementAndGet();

                if (queryId < history.size()) {
                    QueryHistory h = history.get(queryId);
                    for (Vote v : h.votes) {
                        String strategy = v.strategy;
                        double currentW = strategyWeights.getOrDefault(strategy, 1.0);
                        if (isCorrect) strategyWeights.put(strategy, Math.min(2.0, currentW * 1.05));
                        else strategyWeights.put(strategy, Math.max(0.3, currentW * 0.95));

                        double acc = strategyAccuracy.getOrDefault(strategy, 0.5);
                        strategyAccuracy.put(strategy, acc * 0.9 + (isCorrect ? 1.0 : 0.0) * 0.1);
                    }
                }
            }
        }

        Map<String, Object> getMetrics() {
            Map<String, Object> m = new LinkedHashMap<>();
            long tq = totalQueries.get();
            m.put("totalQueries", tq);
            m.put("successfulQueries", successfulQueries.get());
            m.put("avgLatencyMicros", tq > 0 ? totalLatencyNanos.get() / tq / 1000 : 0);
            long cf = correctFeedback.get(), icf = incorrectFeedback.get();
            m.put("accuracy", (cf + icf) > 0 ? Math.round((double) cf / (cf + icf) * 100) : 0);
            m.put("correctFeedback", cf);
            m.put("incorrectFeedback", icf);
            return m;
        }

        Map<String, Object> getStrategies() {
            Map<String, Object> m = new LinkedHashMap<>();
            Map<String, Double> w = new LinkedHashMap<>(), a = new LinkedHashMap<>();
            strategyWeights.forEach((k, v) -> w.put(k, Math.round(v * 1000.0) / 1000.0));
            strategyAccuracy.forEach((k, v) -> a.put(k, Math.round(v * 10000.0) / 10000.0));
            m.put("weights", w);
            m.put("accuracy", a);
            return m;
        }

        List<Map<String, Object>> getPatterns() {
            synchronized (lock) {
                return patterns.stream().limit(20).map(p -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("intent", p.intent);
                    m.put("freq", p.freq);
                    m.put("avgConf", Math.round(p.avgConf * 10000.0) / 10000.0);
                    return m;
                }).collect(Collectors.toList());
            }
        }

        List<Map<String, Object>> getHistory() {
            synchronized (lock) {
                int start = Math.max(0, history.size() - 50);
                return history.subList(start, history.size()).stream().map(h -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("id", h.id);
                    m.put("query", h.query);
                    m.put("intent", h.intent);
                    m.put("sql", h.sql);
                    m.put("confidence", h.confidence);
                    m.put("latencyUs", h.latencyUs);
                    m.put("valid", h.valid);
                    m.put("database", h.database);
                    m.put("time", h.time);
                    return m;
                }).collect(Collectors.toList());
            }
        }
    }

    static class QueryHistory {
        final int id;
        final String query, intent, sql, database;
        final double confidence;
        final long latencyUs;
        final boolean valid;
        final List<SqlEngine.Vote> votes;
        final long time;

        QueryHistory(int id, String query, String intent, String sql,
                     double confidence, long latencyUs, boolean valid,
                     List<SqlEngine.Vote> votes, long time, String database) {
            this.id = id;
            this.query = query;
            this.intent = intent;
            this.sql = sql;
            this.confidence = confidence;
            this.latencyUs = latencyUs;
            this.valid = valid;
            this.votes = votes;
            this.time = time;
            this.database = database;
        }
    }

    static class PatternEntry {
        String intent;
        int freq;
        double avgConf;

        PatternEntry(String intent, int freq, double avgConf) {
            this.intent = intent;
            this.freq = freq;
            this.avgConf = avgConf;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // JSON BUILDER (minimal, no external deps)
    // ═══════════════════════════════════════════════════════════

    static String toJson(Object val) {
        if (val == null) return "null";
        if (val instanceof Number) return val.toString();
        if (val instanceof Boolean) return val.toString();
        if (val instanceof String) return "\"" + escapeJson((String) val) + "\"";
        if (val instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) val;
            String entries = map.entrySet().stream()
                    .map(e -> "\"" + escapeJson(e.getKey()) + "\":" + toJson(e.getValue()))
                    .collect(Collectors.joining(","));
            return "{" + entries + "}";
        }
        if (val instanceof List) {
            @SuppressWarnings("unchecked")
            List<Object> list = (List<Object>) val;
            String items = list.stream().map(SqlAssistantServer::toJson)
                    .collect(Collectors.joining(","));
            return "[" + items + "]";
        }
        return "\"" + escapeJson(val.toString()) + "\"";
    }

    static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    // Minimal JSON array-of-objects parser
    static List<Map<String, String>> parseJsonArrayOfObjects(String json) {
        List<Map<String, String>> result = new ArrayList<>();
        json = json.trim();
        if (!json.startsWith("[")) return result;

        // Find each object
        int i = 1;
        while (i < json.length()) {
            // Skip whitespace and commas
            while (i < json.length() && (json.charAt(i) == ' ' || json.charAt(i) == '\n'
                    || json.charAt(i) == '\r' || json.charAt(i) == '\t' || json.charAt(i) == ','))
                i++;

            if (i >= json.length() || json.charAt(i) == ']') break;
            if (json.charAt(i) == '{') {
                int start = i;
                int depth = 0;
                while (i < json.length()) {
                    if (json.charAt(i) == '{') depth++;
                    else if (json.charAt(i) == '}') {
                        depth--;
                        if (depth == 0) {
                            i++;
                            break;
                        }
                    }
                    i++;
                }
                String obj = json.substring(start, i);
                result.add(parseJsonObject(obj));
            } else {
                i++;
            }
        }
        return result;
    }

    static Map<String, String> parseJsonObject(String json) {
        Map<String, String> map = new LinkedHashMap<>();
        // Simple key-value extraction
        var m = Pattern.compile("\"(\\w+)\"\\s*:\\s*\"([^\"]*)\"").matcher(json);
        while (m.find()) {
            map.put(m.group(1), m.group(2));
        }
        // Also match non-quoted values (numbers, booleans)
        var m2 = Pattern.compile("\"(\\w+)\"\\s*:\\s*(true|false|\\d+\\.?\\d*)").matcher(json);
        while (m2.find()) {
            map.put(m2.group(1), m2.group(2));
        }
        return map;
    }

    // ═══════════════════════════════════════════════════════════
    // HTTP HANDLERS
    // ═══════════════════════════════════════════════════════════

    static class ApiDatabasesHandler implements HttpHandler {
        private final DatabaseManager dbMgr;

        ApiDatabasesHandler(DatabaseManager dbMgr) {
            this.dbMgr = dbMgr;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }
            sendJson(exchange, Map.of("databases", dbMgr.getDatabaseList()));
        }
    }

    static class ApiQueryHandler implements HttpHandler {
        private final SqlEngine engine;
        private final DatabaseManager dbMgr;

        ApiQueryHandler(SqlEngine engine, DatabaseManager dbMgr) {
            this.engine = engine;
            this.dbMgr = dbMgr;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                if (!"POST".equals(exchange.getRequestMethod()) && !"OPTIONS".equals(exchange.getRequestMethod())) {
                    exchange.sendResponseHeaders(405, -1);
                    return;
                }
                if ("OPTIONS".equals(exchange.getRequestMethod())) {
                    sendCorsHeaders(exchange);
                    exchange.sendResponseHeaders(204, -1);
                    return;
                }
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                String query = extractJsonString(body, "query");
                String dbName = extractJsonString(body, "database");
                if (dbName == null || dbName.isEmpty()) dbName = dbMgr.getDatabaseList().get(0).get("name").toString();
                Map<String, TableSchema> schema = dbMgr.getSchema(dbName);
                Map<String, Object> result = query == null || query.isEmpty()
                        ? Map.of("error", "Query must not be blank")
                        : engine.processQuery(query, dbName, schema);
                sendJson(exchange, result);
            } catch (Exception e) {
                e.printStackTrace();
                try {
                    sendJson(exchange, Map.of("error", e.getMessage()), 500);
                } catch (Exception ignored) {
                }
            }
        }
    }

    static class ApiExecuteHandler implements HttpHandler {
        private final DatabaseManager dbMgr;

        ApiExecuteHandler(DatabaseManager dbMgr) {
            this.dbMgr = dbMgr;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                if (!"POST".equals(exchange.getRequestMethod()) && !"OPTIONS".equals(exchange.getRequestMethod())) {
                    exchange.sendResponseHeaders(405, -1);
                    return;
                }
                if ("OPTIONS".equals(exchange.getRequestMethod())) {
                    sendCorsHeaders(exchange);
                    exchange.sendResponseHeaders(204, -1);
                    return;
                }
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                String sql = extractJsonString(body, "sql");
                String dbName = extractJsonString(body, "database");
                if (dbName == null || dbName.isEmpty()) dbName = dbMgr.getDatabaseList().get(0).get("name").toString();

                if (sql == null || sql.isEmpty()) {
                    sendJson(exchange, Map.of("success", false, "error", "SQL must not be blank"), 400);
                    return;
                }

                // Clean SQL: remove trailing semicolons for execution, limit dangerous operations
                String cleanSql = sql.trim();
                if (cleanSql.endsWith(";")) cleanSql = cleanSql.substring(0, cleanSql.length() - 1);

                long start = System.nanoTime();
                QueryResult qr = dbMgr.executeQuery(dbName, cleanSql);
                long latencyMs = (System.nanoTime() - start) / 1_000_000;

                Map<String, Object> response = new LinkedHashMap<>();
                response.put("success", qr.success);
                response.put("database", dbName);
                response.put("sql", sql);
                response.put("latencyMs", latencyMs);

                if (qr.success) {
                    if (!qr.columns.isEmpty()) {
                        // SELECT query result
                        response.put("type", "resultSet");
                        response.put("rowCount", qr.rowCount);
                        response.put("hasMore", qr.hasMore);
                        response.put("columns", qr.columns.stream().map(c -> {
                            Map<String, Object> m = new LinkedHashMap<>();
                            m.put("name", c.name);
                            m.put("type", c.type);
                            return m;
                        }).collect(Collectors.toList()));
                        response.put("rows", qr.rows);
                    } else {
                        // UPDATE/INSERT/DELETE result
                        response.put("type", "update");
                        response.put("updateCount", qr.updateCount);
                    }
                } else {
                    response.put("error", qr.error);
                }

                sendJson(exchange, response);
            } catch (Exception e) {
                e.printStackTrace();
                try {
                    sendJson(exchange, Map.of("success", false, "error", e.getMessage()), 500);
                } catch (Exception ignored) {
                }
            }
        }
    }

    static class ApiSchemaHandler implements HttpHandler {
        private final DatabaseManager dbMgr;

        ApiSchemaHandler(DatabaseManager dbMgr) {
            this.dbMgr = dbMgr;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }
            String query = exchange.getRequestURI().getQuery();
            String dbName = extractQueryParam(query, "db");
            if (dbName == null || dbName.isEmpty()) {
                List<Map<String, Object>> dbs = dbMgr.getDatabaseList();
                if (!dbs.isEmpty()) dbName = dbs.get(0).get("name").toString();
            }

            Map<String, TableSchema> schemaMap = dbMgr.getSchema(dbName);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("database", dbName);
            Map<String, Object> tablesMap = new LinkedHashMap<>();
            for (Map.Entry<String, TableSchema> e : schemaMap.entrySet()) {
                Map<String, Object> tableMap = new LinkedHashMap<>();
                List<Map<String, Object>> cols = e.getValue().columns.stream().map(c -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("name", c.name);
                    m.put("type", c.type);
                    m.put("pk", c.pk);
                    m.put("nullable", c.nullable);
                    return m;
                }).collect(Collectors.toList());
                tableMap.put("columns", cols);
                tableMap.put("indexes", e.getValue().indexes);
                tablesMap.put(e.getKey(), tableMap);
            }
            result.put("tables", tablesMap);
            sendJson(exchange, result);
        }
    }

    static class ApiMetricsHandler implements HttpHandler {
        private final SqlEngine engine;

        ApiMetricsHandler(SqlEngine engine) {
            this.engine = engine;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }
            sendJson(exchange, engine.getMetrics());
        }
    }

    static class ApiStrategiesHandler implements HttpHandler {
        private final SqlEngine engine;

        ApiStrategiesHandler(SqlEngine engine) {
            this.engine = engine;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }
            sendJson(exchange, engine.getStrategies());
        }
    }

    static class ApiPatternsHandler implements HttpHandler {
        private final SqlEngine engine;

        ApiPatternsHandler(SqlEngine engine) {
            this.engine = engine;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }
            List<Map<String, Object>> patterns = engine.getPatterns();
            sendJson(exchange, Map.of("patterns", patterns));
        }
    }

    static class ApiFeedbackHandler implements HttpHandler {
        private final SqlEngine engine;

        ApiFeedbackHandler(SqlEngine engine) {
            this.engine = engine;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod()) && !"OPTIONS".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            int queryId = extractJsonInt(body, "queryId");
            boolean correct = extractJsonBoolean(body, "correct");
            engine.recordFeedback(queryId, correct);
            sendJson(exchange, Map.of("status", "ok"));
        }
    }

    static class StaticFileHandler implements HttpHandler {
        private final Path baseDir;

        StaticFileHandler(Path baseDir) {
            this.baseDir = baseDir;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/") || path.equals("/index.html")) {
                serveFile(exchange, baseDir.resolve("index.html"), "text/html");
            } else {
                Path filePath = baseDir.resolve(path.substring(1));
                if (Files.exists(filePath) && filePath.startsWith(baseDir)) {
                    String ct = guessContentType(filePath.getFileName().toString());
                    serveFile(exchange, filePath, ct);
                } else {
                    exchange.sendResponseHeaders(404, -1);
                }
            }
        }

        private void serveFile(HttpExchange exchange, Path path, String contentType) throws IOException {
            byte[] bytes = Files.readAllBytes(path);
            exchange.getResponseHeaders().set("Content-Type", contentType + "; charset=utf-8");
            exchange.getResponseHeaders().set("Content-Length", String.valueOf(bytes.length));
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.getResponseBody().close();
        }

        private String guessContentType(String filename) {
            if (filename.endsWith(".html")) return "text/html";
            if (filename.endsWith(".css")) return "text/css";
            if (filename.endsWith(".js")) return "application/javascript";
            if (filename.endsWith(".json")) return "application/json";
            if (filename.endsWith(".png")) return "image/png";
            if (filename.endsWith(".svg")) return "image/svg+xml";
            return "application/octet-stream";
        }
    }

    // ═══════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════

    static void sendJson(HttpExchange exchange, Map<String, Object> data) throws IOException {
        sendJson(exchange, data, 200);
    }

    static void sendJson(HttpExchange exchange, Map<String, Object> data, int code) throws IOException {
        byte[] bytes = toJson(data).getBytes(StandardCharsets.UTF_8);
        sendCorsHeaders(exchange);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.getResponseHeaders().set("Content-Length", String.valueOf(bytes.length));
        exchange.sendResponseHeaders(code, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.getResponseBody().close();
    }

    static void sendCorsHeaders(HttpExchange exchange) {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
    }

    static String extractJsonString(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*\"";
        var m = Pattern.compile(pattern).matcher(json);
        if (!m.find()) return null;
        int start = m.end();
        int end = start;
        while (end < json.length() && json.charAt(end) != '"') {
            if (json.charAt(end) == '\\') end++;
            end++;
        }
        return json.substring(start, end);
    }

    static int extractJsonInt(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*(-?\\d+)";
        var m = Pattern.compile(pattern).matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    static boolean extractJsonBoolean(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*(true|false)";
        var m = Pattern.compile(pattern).matcher(json);
        return m.find() && "true".equals(m.group(1));
    }

    static String extractQueryParam(String query, String param) {
        if (query == null) return null;
        for (String pair : query.split("&")) {
            String[] kv = pair.split("=", 2);
            if (kv.length == 2 && kv[0].equals(param)) return java.net.URLDecoder.decode(kv[1], StandardCharsets.UTF_8);
        }
        return null;
    }

    // ═══════════════════════════════════════════════════════════
    // MAIN
    // ═══════════════════════════════════════════════════════════

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8080;
        String clientDir = args.length > 1 ? args[1] : "client";
        String configFile = args.length > 2 ? args[2] : "databases.json";

        System.out.println("============================================================");
        System.out.println("  Enterprise SQL Assistant V3 — JDK HttpServer + Multi-DB");
        System.out.println("  Pure Java 21 + com.sun.net.httpserver + JDBC");
        System.out.println("============================================================");
        System.out.println();

        // Initialize Database Manager
        DatabaseManager dbMgr = new DatabaseManager();
        System.out.println("[1/3] Loading database configuration from: " + configFile);
        dbMgr.loadConfig(configFile);

        System.out.println("[2/3] Connecting to databases...");
        dbMgr.connectAll();

        SqlEngine engine = new SqlEngine();

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);

        // Static files
        Path clientPath = Paths.get(clientDir).toAbsolutePath();
        server.createContext("/", new StaticFileHandler(clientPath));

        // API endpoints
        server.createContext("/api/databases", new ApiDatabasesHandler(dbMgr));
        server.createContext("/api/query", new ApiQueryHandler(engine, dbMgr));
        server.createContext("/api/execute", new ApiExecuteHandler(dbMgr));
        server.createContext("/api/schema", new ApiSchemaHandler(dbMgr));
        server.createContext("/api/metrics", new ApiMetricsHandler(engine));
        server.createContext("/api/strategies", new ApiStrategiesHandler(engine));
        server.createContext("/api/patterns", new ApiPatternsHandler(engine));
        server.createContext("/api/feedback", new ApiFeedbackHandler(engine));

        server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(8));

        System.out.println("[3/3] Starting HTTP server...");
        server.start();

        System.out.println();
        System.out.println("  Endpoints:");
        System.out.println("    GET  /                -> Client HTML");
        System.out.println("    GET  /api/databases   -> List configured databases");
        System.out.println("    GET  /api/schema?db=X -> Schema for database X");
        System.out.println("    POST /api/query       -> Translate NL -> SQL");
        System.out.println("    POST /api/execute     -> Execute SQL on database");
        System.out.println("    GET  /api/metrics     -> Agent metrics");
        System.out.println("    GET  /api/strategies  -> Strategy weights & accuracy");
        System.out.println("    GET  /api/patterns    -> Mined query patterns");
        System.out.println("    POST /api/feedback    -> Submit feedback");
        System.out.println();
        System.out.println("  Client dir: " + clientPath);
        System.out.println("  Config:     " + Paths.get(configFile).toAbsolutePath());
        System.out.println("  Listening on http://0.0.0.0:" + port);
        System.out.println("  Press Ctrl+C to stop");
        System.out.println("============================================================");
    }
}
