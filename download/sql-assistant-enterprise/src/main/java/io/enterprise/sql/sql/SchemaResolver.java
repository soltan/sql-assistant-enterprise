package io.enterprise.sql.sql;

import io.enterprise.sql.config.AssistantConfig;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Schema resolver — provides table/column metadata for SQL generation.
 * In-memory, deterministic, zero-latency schema lookups.
 *
 * Schema definitions are registered at startup and remain immutable during runtime.
 * Thread-safe via ConcurrentHashMap with copy-on-read semantics.
 *
 * Cloud-native: schema can be loaded from environment variables, config maps,
 * or database introspection at startup time.
 */
public class SchemaResolver {

    /** Table definition: name → column info */
    private final ConcurrentHashMap<String, TableSchema> schemas;

    /** Column index: table.column → column definition */
    private final ConcurrentHashMap<String, ColumnSchema> columnIndex;

    public SchemaResolver() {
        this.schemas = new ConcurrentHashMap<>();
        this.columnIndex = new ConcurrentHashMap<>();
    }

    /**
     * Register a table schema.
     */
    public void registerTable(TableSchema schema) {
        schemas.put(schema.tableName(), schema);
        for (ColumnSchema col : schema.columns()) {
            String key = schema.tableName() + "." + col.name();
            columnIndex.put(key, col);
        }
    }

    /**
     * Get the schema for a table.
     */
    public Optional<TableSchema> getTable(String tableName) {
        return Optional.ofNullable(schemas.get(tableName));
    }

    /**
     * Get a specific column definition.
     */
    public Optional<ColumnSchema> getColumn(String tableName, String columnName) {
        return Optional.ofNullable(columnIndex.get(tableName + "." + columnName));
    }

    /**
     * List all registered table names.
     */
    public Set<String> tableNames() {
        return Collections.unmodifiableSet(schemas.keySet());
    }

    /**
     * Resolve a potentially ambiguous table name.
     * Uses prefix matching and similarity scoring.
     */
    public Optional<String> resolveTableName(String partial) {
        // Exact match first
        if (schemas.containsKey(partial)) {
            return Optional.of(partial);
        }

        // Case-insensitive match
        String lower = partial.toLowerCase(Locale.ROOT);
        for (String name : schemas.keySet()) {
            if (name.toLowerCase(Locale.ROOT).equals(lower)) {
                return Optional.of(name);
            }
        }

        // Prefix match
        for (String name : schemas.keySet()) {
            if (name.toLowerCase(Locale.ROOT).startsWith(lower)) {
                return Optional.of(name);
            }
        }

        return Optional.empty();
    }

    /**
     * Load default demo schemas.
     */
    public void loadDefaults() {
        // Users table
        registerTable(new TableSchema("users", List.of(
            new ColumnSchema("id", "BIGINT", false, true),
            new ColumnSchema("username", "VARCHAR(255)", false, false),
            new ColumnSchema("email", "VARCHAR(512)", false, false),
            new ColumnSchema("created_at", "TIMESTAMP", true, false),
            new ColumnSchema("active", "BOOLEAN", true, false)
        ), List.of("idx_users_username", "idx_users_email")));

        // Orders table
        registerTable(new TableSchema("orders", List.of(
            new ColumnSchema("id", "BIGINT", false, true),
            new ColumnSchema("user_id", "BIGINT", false, false),
            new ColumnSchema("product_id", "BIGINT", false, false),
            new ColumnSchema("quantity", "INTEGER", false, false),
            new ColumnSchema("total_price", "DECIMAL(10,2)", false, false),
            new ColumnSchema("order_date", "TIMESTAMP", false, false),
            new ColumnSchema("status", "VARCHAR(50)", false, false)
        ), List.of("idx_orders_user_id", "idx_orders_status")));

        // Products table
        registerTable(new TableSchema("products", List.of(
            new ColumnSchema("id", "BIGINT", false, true),
            new ColumnSchema("name", "VARCHAR(512)", false, false),
            new ColumnSchema("description", "TEXT", true, false),
            new ColumnSchema("price", "DECIMAL(10,2)", false, false),
            new ColumnSchema("category", "VARCHAR(100)", true, false),
            new ColumnSchema("stock", "INTEGER", false, false)
        ), List.of("idx_products_category")));
    }

    /**
     * Table schema record.
     */
    public record TableSchema(
        String tableName,
        List<ColumnSchema> columns,
        List<String> indexes
    ) {
        public TableSchema {
            columns = List.copyOf(columns);
            indexes = List.copyOf(indexes);
        }

        /**
         * Get column names only.
         */
        public List<String> columnNames() {
            return columns.stream().map(ColumnSchema::name).toList();
        }

        /**
         * Get primary key columns.
         */
        public List<ColumnSchema> primaryKeys() {
            return columns.stream().filter(ColumnSchema::primaryKey).toList();
        }
    }

    /**
     * Column schema record.
     */
    public record ColumnSchema(
        String name,
        String dataType,
        boolean nullable,
        boolean primaryKey
    ) {}
}
