import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.*;
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
 * Enterprise SQL Assistant V5 — JDK HttpServer + Multi-DB + BM25 + SQL Templates
 * ====================================================
 * Pure Java server using com.sun.net.httpserver.HttpServer + JDBC.
 * No framework, no external deps (except JDBC drivers) — just the JDK + H2.
 *
 * Features:
 *   - Multi-database support via JDBC (configured in databases.json)
 *   - Dynamic schema discovery at startup via DatabaseMetaData
 *   - SQL query execution with result sets returned as JSON tables
 *   - 7-strategy ensemble resolver (weighted voting) + BM25 + SQL Templates
 *   - BM25 (Okapi) ranking for template matching & schema relevance
 *   - COMMENT ON TABLE/COLUMN for rich schema documentation
 *   - Adaptive feedback system
 *
 * Architecture:
 *   1. PERCEIVE  — Normalize input + BM25 tokenization
 *   2. REASON    — 7-strategy ensemble resolver (weighted voting + BM25 relevance + template matching)
 *   3. ACT       — Generate SQL from resolved intent (template-first, BM25-ranked)
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
 *   GET  /api/templates   → Available SQL templates
 */
public class SqlAssistantServer {

    // ═══════════════════════════════════════════════════════════
    // DATABASE MANAGER — Multi-DB JDBC Connections + Schema Discovery
    // ═══════════════════════════════════════════════════════════

    static class DatabaseConfig {
        String name, description, driver, url, user, password;
        DatabaseConfig(String name, String description, String driver,
                       String url, String user, String password) {
            this.name = name; this.description = description; this.driver = driver;
            this.url = url; this.user = user; this.password = password;
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
                    e.getOrDefault("name", "unknown"),
                    e.getOrDefault("description", ""),
                    e.getOrDefault("driver", "org.h2.Driver"),
                    e.getOrDefault("url", ""),
                    e.getOrDefault("user", "sa"),
                    e.getOrDefault("password", "")
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

                    // Create table structures (no sample data)
                    createTables(cfg.name, conn);

                    // Discover schema dynamically
                    Map<String, TableSchema> dbSchema = discoverSchema(conn);
                    schemas.put(cfg.name, dbSchema);
                    System.out.println("       Discovered " + dbSchema.size() + " tables");
                } catch (Exception ex) {
                    System.err.println("  [FAIL] " + e.getKey() + ": " + ex.getMessage());
                }
            }
        }

        private Map<String, TableSchema> discoverSchema(Connection conn) throws SQLException {
            Map<String, TableSchema> result = new LinkedHashMap<>();
            DatabaseMetaData meta = conn.getMetaData();

            // Get all tables (filter out H2 system tables)
            try (ResultSet tables = meta.getTables(null, "PUBLIC", "%", new String[]{"TABLE"})) {
                while (tables.next()) {
                    String tableName = tables.getString("TABLE_NAME");
                    // Skip H2 system tables
                    if (tableName.startsWith("INFORMATION_SCHEMA") || tableName.startsWith("SYSTEM_")
                        || tableName.equals("DATABASECHANGELOG") || tableName.equals("DATABASECHANGELOGLOCK"))
                        continue;
                    String tableRemark = tables.getString("REMARKS");
                    if (tableRemark == null) tableRemark = "";
                    List<ColumnDef> columns = new ArrayList<>();
                    List<String> indexes = new ArrayList<>();

                    // Get columns (specify PUBLIC schema to avoid H2 system columns)
                    try (ResultSet cols = meta.getColumns(null, "PUBLIC", tableName, "%")) {
                        while (cols.next()) {
                            String colName = cols.getString("COLUMN_NAME");
                            String colType = cols.getString("TYPE_NAME");
                            int colSize = cols.getInt("COLUMN_SIZE");
                            String nullable = cols.getString("IS_NULLABLE");
                            String isAutoInc = cols.getString("IS_AUTOINCREMENT");
                            String colRemark = cols.getString("REMARKS");
                            if (colRemark == null) colRemark = "";

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
                                "YES".equalsIgnoreCase(nullable), colRemark));
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
                            ? new ColumnDef(c.name, c.type, true, c.nullable, c.comment) : c)
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

                    result.put(tableName.toLowerCase(), new TableSchema(tableName, updatedCols, indexes, tableRemark));
                }
            }
            return result;
        }

        private void createTables(String dbName, Connection conn) throws SQLException {
            Statement stmt = conn.createStatement();
            switch (dbName) {
                case "E-Commerce":
                    stmt.execute("CREATE TABLE IF NOT EXISTS users (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "username VARCHAR(255) NOT NULL, " +
                        "email VARCHAR(512) NOT NULL, " +
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                        "active BOOLEAN DEFAULT TRUE)");
                    stmt.execute("COMMENT ON TABLE users IS 'Utilisateurs inscrits sur la plateforme e-commerce'");
                    stmt.execute("COMMENT ON COLUMN users.id IS 'Identifiant unique de l utilisateur'");
                    stmt.execute("COMMENT ON COLUMN users.username IS 'Nom d utilisateur unique'");
                    stmt.execute("COMMENT ON COLUMN users.email IS 'Adresse email de l utilisateur'");
                    stmt.execute("COMMENT ON COLUMN users.created_at IS 'Date de creation du compte'");
                    stmt.execute("COMMENT ON COLUMN users.active IS 'Indique si le compte est actif'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS products (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "name VARCHAR(512) NOT NULL, " +
                        "description TEXT, " +
                        "price DECIMAL(10,2) NOT NULL, " +
                        "category VARCHAR(100), " +
                        "stock INTEGER NOT NULL DEFAULT 0)");
                    stmt.execute("COMMENT ON TABLE products IS 'Catalogue des produits disponibles'");
                    stmt.execute("COMMENT ON COLUMN products.id IS 'Identifiant unique du produit'");
                    stmt.execute("COMMENT ON COLUMN products.name IS 'Nom du produit'");
                    stmt.execute("COMMENT ON COLUMN products.description IS 'Description detaillee du produit'");
                    stmt.execute("COMMENT ON COLUMN products.price IS 'Prix unitaire du produit'");
                    stmt.execute("COMMENT ON COLUMN products.category IS 'Categorie du produit'");
                    stmt.execute("COMMENT ON COLUMN products.stock IS 'Quantite en stock'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS orders (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "user_id BIGINT NOT NULL, " +
                        "product_id BIGINT NOT NULL, " +
                        "quantity INTEGER NOT NULL, " +
                        "total_price DECIMAL(10,2) NOT NULL, " +
                        "order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                        "status VARCHAR(50) DEFAULT 'pending')");
                    stmt.execute("COMMENT ON TABLE orders IS 'Commandes passees par les utilisateurs'");
                    stmt.execute("COMMENT ON COLUMN orders.id IS 'Identifiant unique de la commande'");
                    stmt.execute("COMMENT ON COLUMN orders.user_id IS 'Reference vers l utilisateur'");
                    stmt.execute("COMMENT ON COLUMN orders.product_id IS 'Reference vers le produit'");
                    stmt.execute("COMMENT ON COLUMN orders.quantity IS 'Quantite commandee'");
                    stmt.execute("COMMENT ON COLUMN orders.total_price IS 'Prix total de la commande'");
                    stmt.execute("COMMENT ON COLUMN orders.order_date IS 'Date de la commande'");
                    stmt.execute("COMMENT ON COLUMN orders.status IS 'Statut de la commande (pending, shipped, delivered, cancelled)'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS order_items (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "order_id BIGINT NOT NULL, " +
                        "product_id BIGINT NOT NULL, " +
                        "quantity INTEGER NOT NULL, " +
                        "unit_price DECIMAL(10,2) NOT NULL)");
                    stmt.execute("COMMENT ON TABLE order_items IS 'Lignes de commande detaillees'");
                    stmt.execute("COMMENT ON COLUMN order_items.order_id IS 'Reference vers la commande'");
                    stmt.execute("COMMENT ON COLUMN order_items.product_id IS 'Reference vers le produit'");
                    stmt.execute("COMMENT ON COLUMN order_items.quantity IS 'Quantite commandee'");
                    stmt.execute("COMMENT ON COLUMN order_items.unit_price IS 'Prix unitaire au moment de la commande'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS categories (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "name VARCHAR(100) NOT NULL, " +
                        "description TEXT, " +
                        "parent_id BIGINT)");
                    stmt.execute("COMMENT ON TABLE categories IS 'Categories de produits hierarchiques'");
                    stmt.execute("COMMENT ON COLUMN categories.name IS 'Nom de la categorie'");
                    stmt.execute("COMMENT ON COLUMN categories.description IS 'Description de la categorie'");
                    stmt.execute("COMMENT ON COLUMN categories.parent_id IS 'Reference vers la categorie parente'");
                    break;

                case "Analytics":
                    stmt.execute("CREATE TABLE IF NOT EXISTS events (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "event_type VARCHAR(100) NOT NULL, " +
                        "user_id BIGINT, " +
                        "session_id VARCHAR(255), " +
                        "page_url VARCHAR(1024), " +
                        "event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                        "properties TEXT)");
                    stmt.execute("COMMENT ON TABLE events IS 'Evenements de tracking utilisateur'");
                    stmt.execute("COMMENT ON COLUMN events.event_type IS 'Type d evenement (click, view, purchase, etc.)'");
                    stmt.execute("COMMENT ON COLUMN events.user_id IS 'Reference vers l utilisateur'");
                    stmt.execute("COMMENT ON COLUMN events.session_id IS 'Identifiant de session'");
                    stmt.execute("COMMENT ON COLUMN events.page_url IS 'URL de la page visitee'");
                    stmt.execute("COMMENT ON COLUMN events.event_time IS 'Horodatage de l evenement'");
                    stmt.execute("COMMENT ON COLUMN events.properties IS 'Proprietes additionnelles au format JSON'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS sessions (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "user_id BIGINT, " +
                        "ip_address VARCHAR(45), " +
                        "browser VARCHAR(100), " +
                        "os VARCHAR(100), " +
                        "country VARCHAR(2), " +
                        "started_at TIMESTAMP NOT NULL, " +
                        "duration_seconds INTEGER, " +
                        "page_views INTEGER DEFAULT 0)");
                    stmt.execute("COMMENT ON TABLE sessions IS 'Sessions de navigation utilisateur'");
                    stmt.execute("COMMENT ON COLUMN sessions.user_id IS 'Reference vers l utilisateur'");
                    stmt.execute("COMMENT ON COLUMN sessions.ip_address IS 'Adresse IP du visiteur'");
                    stmt.execute("COMMENT ON COLUMN sessions.browser IS 'Navigateur utilise'");
                    stmt.execute("COMMENT ON COLUMN sessions.os IS 'Systeme d exploitation'");
                    stmt.execute("COMMENT ON COLUMN sessions.country IS 'Code pays ISO 2 lettres'");
                    stmt.execute("COMMENT ON COLUMN sessions.started_at IS 'Debut de la session'");
                    stmt.execute("COMMENT ON COLUMN sessions.duration_seconds IS 'Duree de la session en secondes'");
                    stmt.execute("COMMENT ON COLUMN sessions.page_views IS 'Nombre de pages vues'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS metrics (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "metric_name VARCHAR(100) NOT NULL, " +
                        "metric_value DECIMAL(12,2) NOT NULL, " +
                        "recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                        "dimension VARCHAR(255))");
                    stmt.execute("COMMENT ON TABLE metrics IS 'Metriques de performance enregistrees'");
                    stmt.execute("COMMENT ON COLUMN metrics.metric_name IS 'Nom de la metrique'");
                    stmt.execute("COMMENT ON COLUMN metrics.metric_value IS 'Valeur de la metrique'");
                    stmt.execute("COMMENT ON COLUMN metrics.recorded_at IS 'Date d enregistrement'");
                    stmt.execute("COMMENT ON COLUMN metrics.dimension IS 'Dimension ou categorie de la metrique'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS page_views (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "url VARCHAR(1024) NOT NULL, " +
                        "title VARCHAR(512), " +
                        "views INTEGER DEFAULT 0, " +
                        "unique_views INTEGER DEFAULT 0, " +
                        "avg_time_seconds DECIMAL(8,2), " +
                        "bounce_rate DECIMAL(5,4))");
                    stmt.execute("COMMENT ON TABLE page_views IS 'Statistiques de vues par page'");
                    stmt.execute("COMMENT ON COLUMN page_views.url IS 'URL de la page'");
                    stmt.execute("COMMENT ON COLUMN page_views.title IS 'Titre de la page'");
                    stmt.execute("COMMENT ON COLUMN page_views.views IS 'Nombre total de vues'");
                    stmt.execute("COMMENT ON COLUMN page_views.unique_views IS 'Nombre de vues uniques'");
                    stmt.execute("COMMENT ON COLUMN page_views.avg_time_seconds IS 'Temps moyen passe sur la page'");
                    stmt.execute("COMMENT ON COLUMN page_views.bounce_rate IS 'Taux de rebond'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS conversions (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "funnel_name VARCHAR(100) NOT NULL, " +
                        "step_name VARCHAR(100) NOT NULL, " +
                        "step_order INTEGER NOT NULL, " +
                        "count INTEGER DEFAULT 0, " +
                        "conversion_rate DECIMAL(5,4), " +
                        "recorded_date DATE NOT NULL)");
                    stmt.execute("COMMENT ON TABLE conversions IS 'Entonnoirs de conversion'");
                    stmt.execute("COMMENT ON COLUMN conversions.funnel_name IS 'Nom de l entonnoir'");
                    stmt.execute("COMMENT ON COLUMN conversions.step_name IS 'Nom de l etape'");
                    stmt.execute("COMMENT ON COLUMN conversions.step_order IS 'Ordre de l etape'");
                    stmt.execute("COMMENT ON COLUMN conversions.count IS 'Nombre de conversions'");
                    stmt.execute("COMMENT ON COLUMN conversions.conversion_rate IS 'Taux de conversion'");
                    stmt.execute("COMMENT ON COLUMN conversions.recorded_date IS 'Date d enregistrement'");
                    break;

                case "RH":
                    stmt.execute("CREATE TABLE IF NOT EXISTS employees (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "first_name VARCHAR(100) NOT NULL, " +
                        "last_name VARCHAR(100) NOT NULL, " +
                        "email VARCHAR(255) NOT NULL, " +
                        "department_id BIGINT, " +
                        "position VARCHAR(100), " +
                        "hire_date DATE NOT NULL, " +
                        "salary DECIMAL(10,2), " +
                        "manager_id BIGINT, " +
                        "active BOOLEAN DEFAULT TRUE)");
                    stmt.execute("COMMENT ON TABLE employees IS 'Employes de l entreprise'");
                    stmt.execute("COMMENT ON COLUMN employees.first_name IS 'Prenom de l employe'");
                    stmt.execute("COMMENT ON COLUMN employees.last_name IS 'Nom de famille de l employe'");
                    stmt.execute("COMMENT ON COLUMN employees.email IS 'Adresse email professionnelle'");
                    stmt.execute("COMMENT ON COLUMN employees.department_id IS 'Reference vers le departement'");
                    stmt.execute("COMMENT ON COLUMN employees.position IS 'Poste occupe'");
                    stmt.execute("COMMENT ON COLUMN employees.hire_date IS 'Date d embauche'");
                    stmt.execute("COMMENT ON COLUMN employees.salary IS 'Salaire actuel'");
                    stmt.execute("COMMENT ON COLUMN employees.manager_id IS 'Reference vers le manager'");
                    stmt.execute("COMMENT ON COLUMN employees.active IS 'Employe actif dans l entreprise'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS departments (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "name VARCHAR(100) NOT NULL, " +
                        "budget DECIMAL(12,2), " +
                        "location VARCHAR(100), " +
                        "head_id BIGINT)");
                    stmt.execute("COMMENT ON TABLE departments IS 'Departements de l entreprise'");
                    stmt.execute("COMMENT ON COLUMN departments.name IS 'Nom du departement'");
                    stmt.execute("COMMENT ON COLUMN departments.budget IS 'Budget annuel du departement'");
                    stmt.execute("COMMENT ON COLUMN departments.location IS 'Localisation du departement'");
                    stmt.execute("COMMENT ON COLUMN departments.head_id IS 'Reference vers le responsable'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS salaries (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "employee_id BIGINT NOT NULL, " +
                        "amount DECIMAL(10,2) NOT NULL, " +
                        "effective_date DATE NOT NULL, " +
                        "change_type VARCHAR(20) NOT NULL, " +
                        "approved_by BIGINT)");
                    stmt.execute("COMMENT ON TABLE salaries IS 'Historique des salaires'");
                    stmt.execute("COMMENT ON COLUMN salaries.employee_id IS 'Reference vers l employe'");
                    stmt.execute("COMMENT ON COLUMN salaries.amount IS 'Montant du salaire'");
                    stmt.execute("COMMENT ON COLUMN salaries.effective_date IS 'Date d effet du salaire'");
                    stmt.execute("COMMENT ON COLUMN salaries.change_type IS 'Type de changement (raise, promotion, adjustment)'");
                    stmt.execute("COMMENT ON COLUMN salaries.approved_by IS 'Reference vers l approbateur'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS leave_requests (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "employee_id BIGINT NOT NULL, " +
                        "leave_type VARCHAR(50) NOT NULL, " +
                        "start_date DATE NOT NULL, " +
                        "end_date DATE NOT NULL, " +
                        "status VARCHAR(20) DEFAULT 'pending', " +
                        "approved_by BIGINT, " +
                        "reason TEXT)");
                    stmt.execute("COMMENT ON TABLE leave_requests IS 'Demandes de conge'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.employee_id IS 'Reference vers l employe'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.leave_type IS 'Type de conge (vacation, sick, personal)'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.start_date IS 'Date de debut du conge'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.end_date IS 'Date de fin du conge'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.status IS 'Statut de la demande (pending, approved, rejected)'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.approved_by IS 'Reference vers l approbateur'");
                    stmt.execute("COMMENT ON COLUMN leave_requests.reason IS 'Raison du conge'");

                    stmt.execute("CREATE TABLE IF NOT EXISTS performance_reviews (" +
                        "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, " +
                        "employee_id BIGINT NOT NULL, " +
                        "reviewer_id BIGINT NOT NULL, " +
                        "period VARCHAR(20) NOT NULL, " +
                        "score DECIMAL(3,1), " +
                        "comments TEXT, " +
                        "review_date DATE NOT NULL)");
                    stmt.execute("COMMENT ON TABLE performance_reviews IS 'Evaluations de performance'");
                    stmt.execute("COMMENT ON COLUMN performance_reviews.employee_id IS 'Reference vers l employe evalue'");
                    stmt.execute("COMMENT ON COLUMN performance_reviews.reviewer_id IS 'Reference vers l evaluateur'");
                    stmt.execute("COMMENT ON COLUMN performance_reviews.period IS 'Periode d evaluation (Q1, Q2, Q3, Q4, annual)'");
                    stmt.execute("COMMENT ON COLUMN performance_reviews.score IS 'Note sur 10'");
                    stmt.execute("COMMENT ON COLUMN performance_reviews.comments IS 'Commentaires de l evaluation'");
                    stmt.execute("COMMENT ON COLUMN performance_reviews.review_date IS 'Date de l evaluation'");
                    break;
            }
            stmt.close();
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
        ResultColumn(String name, String type) { this.name = name; this.type = type; }
    }

    // ═══════════════════════════════════════════════════════════
    // SCHEMA DATA CLASSES
    // ═══════════════════════════════════════════════════════════

    record ColumnDef(String name, String type, boolean pk, boolean nullable, String comment) {}
    record TableSchema(String tableName, List<ColumnDef> columns, List<String> indexes, String comment) {}

    // ═══════════════════════════════════════════════════════════
    // BM25 RANKER — Okapi BM25 for Template & Schema Relevance
    // ═══════════════════════════════════════════════════════════

    /**
     * Pure Java BM25 (Okapi BM25) implementation — no external dependencies.
     *
     * BM25 scoring: score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
     *
     * Where:
     *   f(qi,D)  = term frequency of qi in document D
     *   |D|      = document length (term count)
     *   avgdl    = average document length across corpus
     *   k1       = term frequency saturation parameter (1.2-2.0 typical)
     *   b        = length normalization parameter (0.75 typical)
     *   IDF(qi)  = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
     *
     * Uses:
     *   1. Rank SQL templates by relevance to user query (template pattern + description as document)
     *   2. Rank tables/columns by relevance using COMMENT ON metadata
     *   3. 7th strategy in ensemble resolver (bm25_relevance)
     *
     * Stop words: French + English common words filtered out for better IDF discrimination.
     */
    static class BM25Ranker {

        private static final double K1 = 1.5;   // Term frequency saturation
        private static final double B = 0.75;    // Length normalization
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
            "chez", "sans", "entre", "vers", "chez", "ce", "cet", "cette",
            "ces", "il", "elle", "ils", "elles", "nous", "vous", "on",
            "je", "tu", "me", "te", "se", "lui", "leur", "y", "en",
            "ne", "pas", "plus", "aussi", "tres", "bien", "tout", "tous",
            "toute", "toutes", "autre", "autres", "meme", "chaque"
        );

        /** Document entry for BM25 index */
        record BM25Document(String id, String intent, List<String> tokens, String source) {
            int length() { return tokens.size(); }
        }

        /** Scored result */
        record BM25Result(String id, String intent, double score, String source) {}

        private final Map<String, BM25Document> documents = new LinkedHashMap<>();
        private final Map<String, Integer> df = new LinkedHashMap<>(); // document frequency per term
        private int totalDocs = 0;
        private double avgDocLength = 0;

        /** Tokenize text: lowercase, split on non-alphanumeric, remove stop words, handle French accents */
        static List<String> tokenize(String text) {
            if (text == null || text.isBlank()) return List.of();
            // Normalize: lowercase, replace accents with unaccented equivalents for matching
            String normalized = text.toLowerCase()
                .replace("é", "e").replace("è", "e").replace("ê", "e").replace("ë", "e")
                .replace("à", "a").replace("â", "a")
                .replace("ù", "u").replace("û", "u")
                .replace("ô", "o").replace("î", "i").replace("ï", "i")
                .replace("ç", "c")
                .replace("æ", "ae").replace("œ", "oe");
            String[] raw = normalized.split("[^a-z0-9]+");
            List<String> tokens = new ArrayList<>();
            for (String t : raw) {
                if (t.length() > 1 && !STOP_WORDS.contains(t)) {
                    tokens.add(t);
                }
            }
            return tokens;
        }

        /** Add a document to the BM25 index */
        void addDocument(String id, String intent, String text, String source) {
            List<String> tokens = tokenize(text);
            BM25Document doc = new BM25Document(id, intent, tokens, source);
            documents.put(id, doc);

            // Update document frequencies
            Set<String> uniqueTerms = new HashSet<>(tokens);
            for (String term : uniqueTerms) {
                df.merge(term, 1, Integer::sum);
            }

            // Update corpus stats
            totalDocs = documents.size();
            long totalLen = documents.values().stream().mapToLong(BM25Document::length).sum();
            avgDocLength = totalDocs > 0 ? (double) totalLen / totalDocs : 0;
        }

        /** Rebuild corpus stats after batch insertion */
        void rebuildStats() {
            totalDocs = documents.size();
            df.clear();
            long totalLen = 0;
            for (BM25Document doc : documents.values()) {
                totalLen += doc.length();
                Set<String> uniqueTerms = new HashSet<>(doc.tokens);
                for (String term : uniqueTerms) {
                    df.merge(term, 1, Integer::sum);
                }
            }
            avgDocLength = totalDocs > 0 ? (double) totalLen / totalDocs : 0;
        }

        /** Compute IDF for a term */
        double idf(String term) {
            int n = df.getOrDefault(term, 0);
            return Math.log((totalDocs - n + 0.5) / (n + 0.5) + 1.0);
        }

        /** Compute BM25 score of a query against a single document */
        double score(List<String> queryTokens, BM25Document doc) {
            double score = 0;
            int docLen = doc.length();
            Map<String, Long> tfMap = doc.tokens.stream()
                .collect(Collectors.groupingBy(t -> t, Collectors.counting()));

            for (String qTerm : queryTokens) {
                long tf = tfMap.getOrDefault(qTerm, 0L);
                if (tf == 0) continue;
                double idfVal = idf(qTerm);
                double numerator = tf * (K1 + 1);
                double denominator = tf + K1 * (1 - B + B * docLen / avgDocLength);
                score += idfVal * numerator / denominator;
            }
            return score;
        }

        /** Rank all documents against a query, return top-K results */
        List<BM25Result> rank(String query, int topK) {
            List<String> queryTokens = tokenize(query);
            if (queryTokens.isEmpty() || documents.isEmpty()) return List.of();

            List<BM25Result> results = new ArrayList<>();
            for (BM25Document doc : documents.values()) {
                double s = score(queryTokens, doc);
                if (s > 0) {
                    results.add(new BM25Result(doc.id, doc.intent, s, doc.source));
                }
            }
            results.sort((a, b) -> Double.compare(b.score, a.score));
            return results.size() > topK ? results.subList(0, topK) : results;
        }

        /** Rank documents and return normalized scores (0-1 range) */
        List<BM25Result> rankNormalized(String query, int topK) {
            List<BM25Result> raw = rank(query, topK);
            if (raw.isEmpty()) return raw;
            double maxScore = raw.get(0).score;
            if (maxScore <= 0) return raw;
            return raw.stream().map(r -> new BM25Result(r.id, r.intent, r.score / maxScore, r.source))
                .collect(Collectors.toList());
        }

        /** Find the best matching intent via BM25 */
        BM25Result findBestIntent(String query) {
            List<BM25Result> results = rankNormalized(query, 1);
            return results.isEmpty() ? null : results.get(0);
        }

        /** Build BM25 index from SQL templates */
        void indexTemplates(SqlTemplates templates) {
            for (SqlTemplates.SqlTemplate t : templates.getAllTemplates()) {
                // Document = pattern + description + intent (rich source for matching)
                String docText = t.pattern.replace(".*", " ").replace("\\b", " ")
                    .replace("|", " ") + " " + t.description + " " + t.intent;
                addDocument("tpl_" + t.description.hashCode(), t.intent, docText, "template");
            }
            rebuildStats();
        }

        /** Build BM25 index from schema metadata (table names + COMMENT ON) */
        void indexSchema(Map<String, TableSchema> schema) {
            for (Map.Entry<String, TableSchema> e : schema.entrySet()) {
                TableSchema ts = e.getValue();
                // Document = table name + table comment + all column names + all column comments
                StringBuilder sb = new StringBuilder();
                sb.append(ts.tableName).append(" ");
                sb.append(ts.comment).append(" ");
                for (ColumnDef col : ts.columns) {
                    sb.append(col.name).append(" ");
                    sb.append(col.comment).append(" ");
                    // Also add type hints for matching (e.g., "price" → DECIMAL, "date" → TIMESTAMP)
                    if (col.type.toUpperCase().contains("DECIMAL") || col.type.toUpperCase().contains("INT")) {
                        sb.append("nombre montant valeur quantite numeric ");
                    }
                    if (col.type.toUpperCase().contains("TIMESTAMP") || col.type.toUpperCase().contains("DATE")) {
                        sb.append("date temps periode horodatage ");
                    }
                    if (col.type.toUpperCase().contains("BOOLEAN")) {
                        sb.append("boolean indicateur drapeau flag ");
                    }
                    if (col.type.toUpperCase().contains("VARCHAR") || col.type.toUpperCase().contains("TEXT")) {
                        sb.append("texte nom libelle description ");
                    }
                }
                addDocument("tbl_" + ts.tableName.toLowerCase(), "", sb.toString(), "schema_table");
            }
            rebuildStats();
        }

        /** Find the best matching table for a query using BM25 over schema metadata */
        String findBestTable(String query, Map<String, TableSchema> schema) {
            List<String> queryTokens = tokenize(query);
            if (queryTokens.isEmpty()) return schema.keySet().iterator().next();

            String bestTable = null;
            double bestScore = -1;

            for (Map.Entry<String, TableSchema> e : schema.entrySet()) {
                TableSchema ts = e.getValue();
                // Build mini-document for this table
                String docText = ts.tableName + " " + ts.comment;
                for (ColumnDef col : ts.columns) {
                    docText += " " + col.name + " " + col.comment;
                }
                List<String> docTokens = tokenize(docText);

                // Compute BM25 score
                Map<String, Long> tfMap = docTokens.stream()
                    .collect(Collectors.groupingBy(t -> t, Collectors.counting()));
                double s = 0;
                int docLen = docTokens.size();
                for (String qTerm : queryTokens) {
                    long tf = tfMap.getOrDefault(qTerm, 0L);
                    if (tf == 0) continue;
                    double idfVal = idf(qTerm);
                    double numerator = tf * (K1 + 1);
                    double denominator = tf + K1 * (1 - B + B * docLen / Math.max(avgDocLength, 1));
                    s += idfVal * numerator / denominator;
                }

                // Bonus: direct name match (table name exactly appears in query)
                String tblLower = ts.tableName.toLowerCase();
                if (query.toLowerCase().contains(tblLower)) {
                    s += 3.0; // Strong bonus for direct table name match
                }

                if (s > bestScore) {
                    bestScore = s;
                    bestTable = e.getKey();
                }
            }

            return bestTable != null ? bestTable : schema.keySet().iterator().next();
        }

        int getDocumentCount() { return totalDocs; }
        double getAvgDocLength() { return avgDocLength; }
    }

    // ═══════════════════════════════════════════════════════════
    // ENGINE — Ensemble Resolver + SQL Generator + BM25 + SQL Templates
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

        private final SqlTemplates sqlTemplates = new SqlTemplates();
        private final BM25Ranker bm25Ranker = new BM25Ranker();
        private Map<String, TableSchema> currentSchema = new LinkedHashMap<>();

        SqlEngine() {
            strategyWeights.put("intent_graph", 1.0);
            strategyWeights.put("lsh_multi_probe", 0.9);
            strategyWeights.put("syntactic_parser", 1.2);
            strategyWeights.put("thesaurus_hash", 0.8);
            strategyWeights.put("regex_rules", 0.6);
            strategyWeights.put("template_match", 1.5);
            strategyWeights.put("bm25_relevance", 1.4);  // BM25 — high weight, close to template_match

            strategyAccuracy.put("intent_graph", 0.88);
            strategyAccuracy.put("lsh_multi_probe", 0.82);
            strategyAccuracy.put("syntactic_parser", 0.92);
            strategyAccuracy.put("thesaurus_hash", 0.78);
            strategyAccuracy.put("regex_rules", 0.70);
            strategyAccuracy.put("template_match", 0.95);
            strategyAccuracy.put("bm25_relevance", 0.93);  // BM25 has high accuracy due to IDF weighting
        }

        /** Initialize BM25 index with schema metadata (call after database connection) */
        void initBM25(Map<String, TableSchema> schema) {
            this.currentSchema = schema;
            bm25Ranker.indexTemplates(sqlTemplates);
            bm25Ranker.indexSchema(schema);
            System.out.println("       BM25 index built: " + bm25Ranker.getDocumentCount() + " documents, avg length: "
                + String.format("%.1f", bm25Ranker.getAvgDocLength()));
        }

        record Vote(String strategy, String intentType, double rawConfidence, double weightedScore) {}

        EnsembleResult resolveIntent(String query, Map<String, TableSchema> schema) {
            String q = query.toLowerCase();
            List<Vote> votes = new ArrayList<>();

            votes.add(resolveWithGraph(q));
            votes.add(resolveWithLSH(q));
            votes.add(resolveWithSyntax(q));
            votes.add(resolveWithThesaurus(q));
            votes.add(resolveWithRegex(q));
            votes.add(resolveWithTemplates(q, schema));
            votes.add(resolveWithBM25(q, schema));

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
                               List<Vote> votes, Map<String, Double> scores) {}

        private Vote resolveWithGraph(String q) {
            String intent = "Unknown"; double conf = 0.1;
            if (match(q, "select|show|get|find|list|display|montre|affiche|cherche|liste") &&
                match(q, "from|de|dans")) { intent = "Select"; conf = 0.85; }
            else if (match(q, "how many|combien|count|nombre")) { intent = "Aggregate"; conf = 0.80; }
            else if (match(q, "total|somme|sum")) { intent = "Aggregate"; conf = 0.75; }
            else if (match(q, "insert|add|ajoute|ajouter|creer un")) { intent = "Insert"; conf = 0.85; }
            else if (match(q, "update|modify|change|modifie|modifier|mettre a jour")) { intent = "Update"; conf = 0.80; }
            else if (match(q, "delete|remove|supprime|supprimer|efface")) { intent = "Delete"; conf = 0.85; }
            else if (match(q, "create table|cre.*table|nouvelle table")) { intent = "CreateTable"; conf = 0.90; }
            else if (match(q, "alter|modifier.*table")) { intent = "AlterTable"; conf = 0.90; }
            else if (match(q, "drop|supprimer.*table")) { intent = "DropTable"; conf = 0.90; }
            else if (match(q, "join|combine|fusionne|joindre|combiner")) { intent = "Join"; conf = 0.70; }
            else if (match(q, "explain|analyse|explique")) { intent = "Explain"; conf = 0.90; }
            else if (match(q, "describe|schema|structure|decris|colonnes")) { intent = "SchemaInfo"; conf = 0.80; }
            return new Vote("intent_graph", intent, conf, conf * strategyWeights.get("intent_graph"));
        }

        private Vote resolveWithLSH(String q) {
            String intent = "Unknown"; double conf = 0.1;
            if (match(q, "select|show|get|affiche|montre|liste|cherche")) { intent = "Select"; conf = 0.80; }
            else if (match(q, "count|combien|nombre|total|somme")) { intent = "Aggregate"; conf = 0.75; }
            else if (match(q, "insert|ajout|add")) { intent = "Insert"; conf = 0.82; }
            else if (match(q, "update|modif|change")) { intent = "Update"; conf = 0.78; }
            else if (match(q, "delete|supprim|remove")) { intent = "Delete"; conf = 0.82; }
            else if (match(q, "join|combin|fusion")) { intent = "Join"; conf = 0.68; }
            return new Vote("lsh_multi_probe", intent, conf, conf * strategyWeights.get("lsh_multi_probe"));
        }

        private Vote resolveWithSyntax(String q) {
            String intent = "Unknown"; double conf = 0.1;
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
                if (verbMap.containsKey(word)) { intent = verbMap.get(word); conf = 0.88; break; }
            }
            if ("Unknown".equals(intent) && match(q, "combien|nombre|count|total|moyenne|average|somme|sum|max|min")) {
                intent = "Aggregate"; conf = 0.85;
            }
            return new Vote("syntactic_parser", intent, conf, conf * strategyWeights.get("syntactic_parser"));
        }

        private Vote resolveWithThesaurus(String q) {
            String intent = "Unknown"; double conf = 0.1;
            Map<String, String> thesMap = new LinkedHashMap<>();
            thesMap.put("utilisateurs", "Select"); thesMap.put("clients", "Select");
            thesMap.put("commandes", "Select"); thesMap.put("produits", "Select");
            thesMap.put("employes", "Select"); thesMap.put("salaires", "Select");
            thesMap.put("donnees", "Select"); thesMap.put("enregistrements", "Select");
            thesMap.put("nombre", "Aggregate"); thesMap.put("total", "Aggregate");
            thesMap.put("moyenne", "Aggregate"); thesMap.put("somme", "Aggregate");
            thesMap.put("ajouter", "Insert"); thesMap.put("inserer", "Insert");
            thesMap.put("modifier", "Update"); thesMap.put("supprimer", "Delete");
            thesMap.put("joindre", "Join"); thesMap.put("combiner", "Join");
            thesMap.put("fusionner", "Join"); thesMap.put("expliquer", "Explain");
            thesMap.put("decrire", "SchemaInfo");
            for (Map.Entry<String, String> e : thesMap.entrySet()) {
                if (q.contains(e.getKey())) { intent = e.getValue(); conf = 0.76; break; }
            }
            return new Vote("thesaurus_hash", intent, conf, conf * strategyWeights.get("thesaurus_hash"));
        }

        private Vote resolveWithRegex(String q) {
            String intent = "Unknown"; double conf = 0.1;
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
                    intent = rule[1]; conf = Double.parseDouble(rule[2]); break;
                }
            }
            return new Vote("regex_rules", intent, conf, conf * strategyWeights.get("regex_rules"));
        }

        private Vote resolveWithTemplates(String q, Map<String, TableSchema> schema) {
            SqlTemplates.TemplateMatch match = sqlTemplates.findBestMatch(q, schema);
            if (match != null) {
                return new Vote("template_match", match.template.intent, match.score, match.score * strategyWeights.get("template_match"));
            }
            return new Vote("template_match", "Unknown", 0.1, 0.1 * strategyWeights.get("template_match"));
        }

        private Vote resolveWithBM25(String q, Map<String, TableSchema> schema) {
            // Re-index BM25 if schema changed (different database selected)
            if (!schema.equals(currentSchema)) {
                initBM25(schema);
            }
            BM25Ranker.BM25Result result = bm25Ranker.findBestIntent(q);
            if (result != null && result.score > 0.1 && !result.intent.isEmpty()) {
                // BM25 normalized score is 0-1, use as confidence directly
                double conf = Math.min(result.score, 0.98);
                return new Vote("bm25_relevance", result.intent, conf, conf * strategyWeights.get("bm25_relevance"));
            }
            return new Vote("bm25_relevance", "Unknown", 0.1, 0.1 * strategyWeights.get("bm25_relevance"));
        }

        private boolean match(String q, String regex) {
            return Pattern.compile("\\b(" + regex + ")\\b").matcher(q).find();
        }

        // ── SQL Generator (schema-aware, template-first) ──

        String generateSQL(String query, String intent, Map<String, TableSchema> schema) {
            // Try template match first (now also BM25-enhanced)
            SqlTemplates.TemplateMatch templateMatch = sqlTemplates.findBestMatch(query.toLowerCase(), schema);
            if (templateMatch != null && templateMatch.score > 0.5) {
                return templateMatch.sql.endsWith(";") ? templateMatch.sql : templateMatch.sql + ";";
            }

            // If no template matched, try BM25 to find best template
            if (!schema.equals(currentSchema)) initBM25(schema);
            BM25Ranker.BM25Result bm25Result = bm25Ranker.findBestIntent(query.toLowerCase());
            if (bm25Result != null && bm25Result.score > 0.4 && !bm25Result.intent.isEmpty()
                && bm25Result.source.equals("template")) {
                // BM25 found a template match — use its intent and try template again
                SqlTemplates.TemplateMatch bm25Template = sqlTemplates.findBestMatch(
                    query.toLowerCase(), schema);
                if (bm25Template != null) {
                    return bm25Template.sql.endsWith(";") ? bm25Template.sql : bm25Template.sql + ";";
                }
            }

            String q = query.toLowerCase();
            switch (intent) {
                case "Select": {
                    // Use BM25 for table resolution (falls back to keyword matching)
                    String table = resolveTableBM25(q, schema);
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
                    String table = resolveTableBM25(q, schema);
                    TableSchema ts = schema.get(table);
                    List<String> colNames = ts != null ? ts.columns.stream()
                        .filter(c -> !c.pk).map(c -> c.name).collect(Collectors.toList())
                        : List.of("name", "value");
                    String vals = colNames.stream().map(c -> ":" + c).collect(Collectors.joining(", "));
                    return "INSERT INTO " + table + " (" + String.join(", ", colNames) + ")\nVALUES (" + vals + ");";
                }
                case "Update": {
                    String table = resolveTableBM25(q, schema);
                    List<String> conds = extractConditions(q, schema.get(table));
                    String where = conds.isEmpty() ? "\nWHERE id = :id" : "\nWHERE " + String.join("\n  AND ", conds);
                    return "UPDATE " + table + "\nSET :column = :value" + where + ";";
                }
                case "Delete": {
                    String table = resolveTableBM25(q, schema);
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
                    String table = resolveTableBM25(q, schema);
                    String fn = "COUNT";
                    if (match(q, "sum|total|somme")) fn = "SUM";
                    else if (match(q, "average|avg|moyenne|mean")) fn = "AVG";
                    else if (match(q, "max|maximum|highest|plus grand")) fn = "MAX";
                    else if (match(q, "min|minimum|lowest|plus petit")) fn = "MIN";
                    var gm = Pattern.compile("\\b(group by|par|by)\\s+(\\w+)").matcher(q);
                    String group = ""; String col = "1";
                    if (gm.find()) { group = "\nGROUP BY " + gm.group(2); col = gm.group(2); }
                    List<String> conds = extractConditions(q, schema.get(table));
                    String where = conds.isEmpty() ? "" : "\nWHERE " + String.join("\n  AND ", conds);
                    return "SELECT " + fn + "(*), " + col + "\nFROM " + table + where + group + ";";
                }
                case "Explain":
                    return "EXPLAIN ANALYZE\nSELECT * FROM " + resolveTableBM25(q, schema) + " WHERE id = 1;";
                case "SchemaInfo": {
                    String table = resolveTableBM25(q, schema);
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

        /**
         * Resolve table using BM25 over COMMENT ON metadata.
         * Falls back to keyword-based resolveTable if BM25 returns no results.
         * BM25 leverages table/column comments for semantic matching:
         *   "salaires" matches employees table via comment "Salaire actuel"
         *   "congés en attente" matches leave_requests via comment "Demandes de conge"
         */
        private String resolveTableBM25(String q, Map<String, TableSchema> schema) {
            if (!schema.equals(currentSchema)) initBM25(schema);
            try {
                String bm25Table = bm25Ranker.findBestTable(q, schema);
                // Validate: BM25 must return a key that exists in schema
                if (bm25Table != null && schema.containsKey(bm25Table)) {
                    return bm25Table;
                }
            } catch (Exception e) {
                // Fallback silently
            }
            // Fallback to keyword-based resolution
            return resolveTable(q, schema);
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

            // Add BM25 ranking details for transparency
            try {
                List<BM25Ranker.BM25Result> bm25Top = bm25Ranker.rankNormalized(normalized, 3);
                List<Map<String, Object>> bm25Details = bm25Top.stream().map(r -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("id", r.id);
                    m.put("intent", r.intent);
                    m.put("score", Math.round(r.score * 10000.0) / 10000.0);
                    m.put("source", r.source);
                    return m;
                }).collect(Collectors.toList());
                response.put("bm25TopResults", bm25Details);
            } catch (Exception e) {
                // BM25 details are optional
            }

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

        List<Map<String, Object>> getTemplateList() {
            return sqlTemplates.getTemplateList();
        }

        // ═══════════════════════════════════════════════════════════
        // SQL TEMPLATES — Template-based SQL generation
        // ═══════════════════════════════════════════════════════════

        static class SqlTemplates {
            record SqlTemplate(String pattern, String intent, String sqlTemplate, String description) {}

            private static final List<SqlTemplate> TEMPLATES = List.of(
                // SELECT templates
                new SqlTemplate("affiche.*utilisateurs|montre.*utilisateurs|liste.*utilisateurs", "Select",
                    "SELECT {{columns}} FROM {{table}} WHERE active = TRUE", "Utilisateurs actifs"),
                new SqlTemplate("tous les produits|affiche.*produits|liste.*produits", "Select",
                    "SELECT {{columns}} FROM {{table}}", "Tous les produits"),
                new SqlTemplate("commandes.*recentes|dernieres commandes", "Select",
                    "SELECT {{columns}} FROM {{table}} ORDER BY order_date DESC", "Commandes recentes"),
                new SqlTemplate("commandes.*statut|commandes.*status", "Select",
                    "SELECT {{columns}} FROM {{table}} WHERE status = :status", "Commandes par statut"),
                new SqlTemplate("employes.*departement|employes.*service", "Select",
                    "SELECT e.*, d.name AS department_name FROM {{table}} e JOIN departments d ON e.department_id = d.id", "Employes avec departement"),
                new SqlTemplate("sessions.*navigateur|sessions.*browser", "Select",
                    "SELECT browser, COUNT(*) AS session_count FROM {{table}} GROUP BY browser ORDER BY session_count DESC", "Sessions par navigateur"),
                new SqlTemplate("evenements.*type|events.*type", "Select",
                    "SELECT event_type, COUNT(*) AS event_count FROM {{table}} GROUP BY event_type ORDER BY event_count DESC", "Evenements par type"),
                new SqlTemplate("conges.*en.*attente|leave.*pending", "Select",
                    "SELECT {{columns}} FROM {{table}} WHERE status = 'pending'", "Conges en attente"),

                // AGGREGATE templates
                new SqlTemplate("nombre.*utilisateurs|combien.*utilisateurs|count.*users", "Aggregate",
                    "SELECT COUNT(*) AS total_users FROM {{table}} WHERE active = TRUE", "Nombre d utilisateurs actifs"),
                new SqlTemplate("nombre.*commandes|combien.*commandes|count.*orders", "Aggregate",
                    "SELECT COUNT(*) AS total_orders FROM {{table}}", "Nombre de commandes"),
                new SqlTemplate("chiffre.*affaires|total.*ventes|revenue|ca", "Aggregate",
                    "SELECT SUM(total_price) AS revenue FROM orders", "Chiffre d affaires total"),
                new SqlTemplate("prix.*moyen|average.*price|moyenne.*prix", "Aggregate",
                    "SELECT AVG(price) AS avg_price FROM {{table}}", "Prix moyen des produits"),
                new SqlTemplate("salaire.*departement|salary.*department", "Aggregate",
                    "SELECT d.name, AVG(e.salary) AS avg_salary FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name", "Salaire par departement"),
                new SqlTemplate("salaire.*moyen|average.*salary|moyenne.*salaire", "Aggregate",
                    "SELECT AVG(salary) AS avg_salary FROM {{table}}", "Salaire moyen"),
                new SqlTemplate("commandes.*par.*statut|orders.*by.*status", "Aggregate",
                    "SELECT status, COUNT(*) AS count FROM {{table}} GROUP BY status", "Commandes par statut"),
                new SqlTemplate("vues.*par.*page|page.*views.*top|pages.*plus.*visitees", "Aggregate",
                    "SELECT url, SUM(views) AS total_views FROM {{table}} GROUP BY url ORDER BY total_views DESC", "Pages les plus visitees"),
                new SqlTemplate("taux.*conversion|conversion.*rate", "Aggregate",
                    "SELECT funnel_name, AVG(conversion_rate) AS avg_conversion FROM {{table}} GROUP BY funnel_name", "Taux de conversion moyen"),

                // JOIN templates
                new SqlTemplate("commandes.*utilisateurs|orders.*users|commandes.*avec.*user", "Join",
                    "SELECT o.*, u.username FROM orders o JOIN users u ON o.user_id = u.id", "Commandes avec utilisateurs"),
                new SqlTemplate("produits.*categories|products.*categories", "Join",
                    "SELECT p.*, c.name AS category_name FROM products p LEFT JOIN categories c ON p.category = c.name", "Produits avec categories"),
                new SqlTemplate("employes.*manager|employees.*manager", "Join",
                    "SELECT e.first_name, e.last_name, m.first_name AS manager_first, m.last_name AS manager_last FROM employees e LEFT JOIN employees m ON e.manager_id = m.id", "Employes avec manager"),
                new SqlTemplate("employes.*salaires|employees.*salaries", "Join",
                    "SELECT e.first_name, e.last_name, s.amount, s.effective_date FROM employees e JOIN salaries s ON e.id = s.employee_id", "Employes avec salaires"),

                // INSERT templates
                new SqlTemplate("ajouter.*utilisateur|creer.*compte|new.*user", "Insert",
                    "INSERT INTO {{table}} (username, email, active) VALUES (:username, :email, TRUE)", "Nouvel utilisateur"),
                new SqlTemplate("ajouter.*produit|new.*product|creer.*produit", "Insert",
                    "INSERT INTO {{table}} (name, description, price, category, stock) VALUES (:name, :description, :price, :category, :stock)", "Nouveau produit"),
                new SqlTemplate("nouvelle.*commande|creer.*commande|new.*order", "Insert",
                    "INSERT INTO {{table}} (user_id, product_id, quantity, total_price, status) VALUES (:user_id, :product_id, :quantity, :total_price, 'pending')", "Nouvelle commande"),

                // UPDATE templates
                new SqlTemplate("desactiver.*utilisateur|deactivate.*user", "Update",
                    "UPDATE {{table}} SET active = FALSE WHERE id = :id", "Desactiver un utilisateur"),
                new SqlTemplate("activer.*utilisateur|activate.*user", "Update",
                    "UPDATE {{table}} SET active = TRUE WHERE id = :id", "Activer un utilisateur"),
                new SqlTemplate("modifier.*stock|update.*stock", "Update",
                    "UPDATE {{table}} SET stock = :stock WHERE id = :id", "Modifier le stock"),
                new SqlTemplate("approuver.*conge|approve.*leave", "Update",
                    "UPDATE {{table}} SET status = 'approved', approved_by = :approved_by WHERE id = :id", "Approuver un conge"),

                // DELETE templates
                new SqlTemplate("supprimer.*utilisateur|delete.*user", "Delete",
                    "DELETE FROM {{table}} WHERE id = :id", "Supprimer un utilisateur"),
                new SqlTemplate("supprimer.*produit|delete.*product", "Delete",
                    "DELETE FROM {{table}} WHERE id = :id", "Supprimer un produit")
            );

            /** Find the best matching template for a query */
            TemplateMatch findBestMatch(String query, Map<String, TableSchema> schema) {
                String q = query.toLowerCase().trim();
                TemplateMatch bestMatch = null;
                double bestScore = 0;

                for (SqlTemplate t : TEMPLATES) {
                    if (Pattern.compile(t.pattern).matcher(q).find()) {
                        double score = computeScore(q, t);
                        if (score > bestScore) {
                            bestScore = score;
                            String table = resolveTableForTemplate(q, t, schema);
                            String columns = resolveColumnsForTemplate(table, schema);
                            String sql = t.sqlTemplate
                                .replace("{{table}}", table)
                                .replace("{{columns}}", columns);
                            bestMatch = new TemplateMatch(t, sql, table, score);
                        }
                    }
                }
                return bestMatch;
            }

            private double computeScore(String q, SqlTemplate t) {
                double score = 0.75; // Base score for any template regex match
                // Higher score for more keyword matches within the pattern alternatives
                // Also prefer more specific (longer) patterns
                String[] alternatives = t.pattern.split("\\|");
                double bestAltScore = 0;
                for (String alt : alternatives) {
                    String[] parts = alt.replace(".*", " ").replace("\\b", " ").trim().split("\\s+");
                    int matchCount = 0;
                    int totalLen = 0;
                    for (String part : parts) {
                        totalLen += part.length();
                        if (part.length() > 2 && q.contains(part.toLowerCase())) matchCount++;
                    }
                    double altScore = 0.05 * matchCount + 0.001 * totalLen; // Bonus for longer/more specific patterns
                    if (altScore > bestAltScore) bestAltScore = altScore;
                }
                score += bestAltScore;
                return Math.min(score, 0.98);
            }

            private String resolveTableForTemplate(String q, SqlTemplate t, Map<String, TableSchema> schema) {
                // Try to find the table based on intent + query keywords
                for (Map.Entry<String, TableSchema> e : schema.entrySet()) {
                    String tbl = e.getKey();
                    if (q.contains(tbl) || q.contains(tbl.endsWith("s") ? tbl.substring(0, tbl.length()-1) : tbl)) {
                        return e.getValue().tableName;
                    }
                }
                // Default table based on common keywords (order matters - most specific first)
                if (q.contains("utilisat") || q.contains("user") || q.contains("compte")) return findTbl(schema, "users");
                if (q.contains("produit") || q.contains("product")) return findTbl(schema, "products");
                if (q.contains("command") || q.contains("order") || q.contains("affaire") || q.contains("vente") || q.contains("ventes") || q.contains("chiffre")) return findTbl(schema, "orders");
                if (q.contains("salar") || q.contains("salaire")) return findTbl(schema, "salaries");
                if (q.contains("employ") || q.contains("personnel")) return findTbl(schema, "employees");
                if (q.contains("depart") || q.contains("service")) return findTbl(schema, "departments");
                if (q.contains("conge") || q.contains("leave")) return findTbl(schema, "leave_requests");
                if (q.contains("perform") || q.contains("evaluation")) return findTbl(schema, "performance_reviews");
                if (q.contains("session")) return findTbl(schema, "sessions");
                if (q.contains("evenement") || q.contains("event")) return findTbl(schema, "events");
                if (q.contains("page") || q.contains("vue")) return findTbl(schema, "page_views");
                if (q.contains("conversion") || q.contains("entonnoir") || q.contains("funnel")) return findTbl(schema, "conversions");
                if (q.contains("metrique") || q.contains("metric")) return findTbl(schema, "metrics");
                if (q.contains("categori")) return findTbl(schema, "categories");
                // Fallback
                return schema.values().iterator().next().tableName;
            }

            private String findTbl(Map<String, TableSchema> schema, String prefix) {
                for (String name : schema.keySet()) {
                    if (name.toLowerCase().startsWith(prefix)) return schema.get(name).tableName;
                }
                return schema.values().iterator().next().tableName;
            }

            private String resolveColumnsForTemplate(String tableName, Map<String, TableSchema> schema) {
                for (TableSchema ts : schema.values()) {
                    if (ts.tableName.equalsIgnoreCase(tableName)) {
                        return ts.columns.stream()
                            .filter(c -> !c.pk)
                            .map(c -> c.name)
                            .limit(8)
                            .collect(Collectors.joining(", "));
                    }
                }
                return "*";
            }

            /** Get all available templates as a list for the client */
            List<Map<String, Object>> getTemplateList() {
                return TEMPLATES.stream().map(t -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("pattern", t.pattern);
                    m.put("intent", t.intent);
                    m.put("description", t.description);
                    m.put("sqlTemplate", t.sqlTemplate);
                    return m;
                }).collect(Collectors.toList());
            }

            /** Expose templates for BM25 indexing */
            List<SqlTemplate> getAllTemplates() {
                return TEMPLATES;
            }

            record TemplateMatch(SqlTemplate template, String sql, String resolvedTable, double score) {}
        }
    }

    static class QueryHistory {
        final int id; final String query, intent, sql, database;
        final double confidence; final long latencyUs; final boolean valid;
        final List<SqlEngine.Vote> votes; final long time;
        QueryHistory(int id, String query, String intent, String sql,
                      double confidence, long latencyUs, boolean valid,
                      List<SqlEngine.Vote> votes, long time, String database) {
            this.id = id; this.query = query; this.intent = intent; this.sql = sql;
            this.confidence = confidence; this.latencyUs = latencyUs; this.valid = valid;
            this.votes = votes; this.time = time; this.database = database;
        }
    }

    static class PatternEntry {
        String intent; int freq; double avgConf;
        PatternEntry(String intent, int freq, double avgConf) {
            this.intent = intent; this.freq = freq; this.avgConf = avgConf;
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
                        if (depth == 0) { i++; break; }
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
        ApiDatabasesHandler(DatabaseManager dbMgr) { this.dbMgr = dbMgr; }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
            }
            sendJson(exchange, Map.of("databases", dbMgr.getDatabaseList()));
        }
    }

    static class ApiQueryHandler implements HttpHandler {
        private final SqlEngine engine;
        private final DatabaseManager dbMgr;
        ApiQueryHandler(SqlEngine engine, DatabaseManager dbMgr) {
            this.engine = engine; this.dbMgr = dbMgr;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                if (!"POST".equals(exchange.getRequestMethod()) && !"OPTIONS".equals(exchange.getRequestMethod())) {
                    exchange.sendResponseHeaders(405, -1); return;
                }
                if ("OPTIONS".equals(exchange.getRequestMethod())) {
                    sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
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
                try { sendJson(exchange, Map.of("error", e.getMessage()), 500); }
                catch (Exception ignored) {}
            }
        }
    }

    static class ApiExecuteHandler implements HttpHandler {
        private final DatabaseManager dbMgr;
        ApiExecuteHandler(DatabaseManager dbMgr) { this.dbMgr = dbMgr; }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                if (!"POST".equals(exchange.getRequestMethod()) && !"OPTIONS".equals(exchange.getRequestMethod())) {
                    exchange.sendResponseHeaders(405, -1); return;
                }
                if ("OPTIONS".equals(exchange.getRequestMethod())) {
                    sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
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
                            m.put("name", c.name); m.put("type", c.type);
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
                try { sendJson(exchange, Map.of("error", e.getMessage()), 500); }
                catch (Exception ignored) {}
            }
        }
    }

    static class ApiSchemaHandler implements HttpHandler {
        private final DatabaseManager dbMgr;
        ApiSchemaHandler(DatabaseManager dbMgr) { this.dbMgr = dbMgr; }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
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
                    m.put("name", c.name); m.put("type", c.type);
                    m.put("pk", c.pk); m.put("nullable", c.nullable);
                    m.put("comment", c.comment);
                    return m;
                }).collect(Collectors.toList());
                tableMap.put("columns", cols);
                tableMap.put("indexes", e.getValue().indexes);
                tableMap.put("comment", e.getValue().comment);
                tablesMap.put(e.getKey(), tableMap);
            }
            result.put("tables", tablesMap);
            sendJson(exchange, result);
        }
    }

    static class ApiMetricsHandler implements HttpHandler {
        private final SqlEngine engine;
        ApiMetricsHandler(SqlEngine engine) { this.engine = engine; }
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
            }
            sendJson(exchange, engine.getMetrics());
        }
    }

    static class ApiStrategiesHandler implements HttpHandler {
        private final SqlEngine engine;
        ApiStrategiesHandler(SqlEngine engine) { this.engine = engine; }
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
            }
            sendJson(exchange, engine.getStrategies());
        }
    }

    static class ApiPatternsHandler implements HttpHandler {
        private final SqlEngine engine;
        ApiPatternsHandler(SqlEngine engine) { this.engine = engine; }
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
            }
            List<Map<String, Object>> patterns = engine.getPatterns();
            sendJson(exchange, Map.of("patterns", patterns));
        }
    }

    static class ApiFeedbackHandler implements HttpHandler {
        private final SqlEngine engine;
        ApiFeedbackHandler(SqlEngine engine) { this.engine = engine; }
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod()) && !"OPTIONS".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1); return;
            }
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
            }
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            int queryId = extractJsonInt(body, "queryId");
            boolean correct = extractJsonBoolean(body, "correct");
            engine.recordFeedback(queryId, correct);
            sendJson(exchange, Map.of("status", "ok"));
        }
    }

    static class ApiTemplatesHandler implements HttpHandler {
        private final SqlEngine engine;
        ApiTemplatesHandler(SqlEngine engine) { this.engine = engine; }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("OPTIONS".equals(exchange.getRequestMethod())) {
                sendCorsHeaders(exchange); exchange.sendResponseHeaders(204, -1); return;
            }
            List<Map<String, Object>> templates = engine.getTemplateList();
            sendJson(exchange, Map.of("templates", templates));
        }
    }

    static class StaticFileHandler implements HttpHandler {
        private final Path baseDir;
        StaticFileHandler(Path baseDir) { this.baseDir = baseDir; }

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
        System.out.println("  Enterprise SQL Assistant V5 — JDK HttpServer + BM25 + Multi-DB");
        System.out.println("  Pure Java 21 + com.sun.net.httpserver + JDBC + Okapi BM25");
        System.out.println("============================================================");
        System.out.println();

        // Initialize Database Manager
        DatabaseManager dbMgr = new DatabaseManager();
        System.out.println("[1/3] Loading database configuration from: " + configFile);
        dbMgr.loadConfig(configFile);

        System.out.println("[2/3] Connecting to databases...");
        dbMgr.connectAll();

        SqlEngine engine = new SqlEngine();

        // Initialize BM25 index with first database's schema
        List<Map<String, Object>> dbList = dbMgr.getDatabaseList();
        if (!dbList.isEmpty()) {
            String firstDb = dbList.get(0).get("name").toString();
            engine.initBM25(dbMgr.getSchema(firstDb));
        }

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
        server.createContext("/api/templates", new ApiTemplatesHandler(engine));

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
        System.out.println("    GET  /api/templates   -> Available SQL templates");
        System.out.println();
        System.out.println("  Client dir: " + clientPath);
        System.out.println("  Config:     " + Paths.get(configFile).toAbsolutePath());
        System.out.println("  Listening on http://0.0.0.0:" + port);
        System.out.println("  Press Ctrl+C to stop");
        System.out.println("============================================================");
    }
}
}
