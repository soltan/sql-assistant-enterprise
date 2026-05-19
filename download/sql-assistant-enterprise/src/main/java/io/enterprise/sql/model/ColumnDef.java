package io.enterprise.sql.model;

/**
 * Column definition record — pure data, immutable.
 * Used in CREATE TABLE and ALTER TABLE intents.
 *
 * @param name     column name
 * @param dataType SQL data type (e.g., "VARCHAR(255)", "INTEGER", "TIMESTAMP")
 * @param nullable whether the column allows NULL
 * @param primaryKey whether this column is part of the primary key
 */
public record ColumnDef(
    String name,
    String dataType,
    boolean nullable,
    boolean primaryKey
) {
    /**
     * Compact constructor with validation.
     */
    public ColumnDef {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Column name must not be blank");
        }
        if (dataType == null || dataType.isBlank()) {
            throw new IllegalArgumentException("Data type must not be blank");
        }
    }

    /**
     * Convenience factory for a non-nullable column.
     */
    public static ColumnDef of(String name, String dataType) {
        return new ColumnDef(name, dataType, false, false);
    }

    /**
     * Convenience factory for a primary key column.
     */
    public static ColumnDef primaryKey(String name, String dataType) {
        return new ColumnDef(name, dataType, false, true);
    }
}
