package io.enterprise.sql.model;

/**
 * Sealed interface hierarchy representing all possible SQL intents.
 * Data Oriented Programming: exhaustive pattern matching guaranteed at compile time.
 * Each variant carries exactly the data it needs — no nulls, no optional abuse.
 */
public sealed interface SqlIntent
    permits SqlIntent.Select, SqlIntent.Insert, SqlIntent.Update, SqlIntent.Delete,
            SqlIntent.CreateTable, SqlIntent.AlterTable, SqlIntent.DropTable,
            SqlIntent.Join, SqlIntent.Aggregate, SqlIntent.Subquery,
            SqlIntent.Explain, SqlIntent.SchemaInfo, SqlIntent.Unknown {

    /**
     * The raw natural language query that produced this intent.
     */
    String rawQuery();

    /**
     * Confidence score [0.0, 1.0] from the intent resolver.
     */
    double confidence();

    /**
     * SELECT intent — the most common query type.
     *
     * @param rawQuery    original user query
     * @param confidence  resolver confidence
     * @param tables      target table names (at least one)
     * @param columns     requested columns (empty = SELECT *)
     * @param whereClause optional filter description
     * @param orderBy     optional sort description
     * @param limit       optional row limit (0 = no limit)
     */
    record Select(
        String rawQuery,
        double confidence,
        java.util.List<String> tables,
        java.util.List<String> columns,
        String whereClause,
        String orderBy,
        int limit
    ) implements SqlIntent {
        public Select {
            tables = java.util.List.copyOf(tables);
            columns = java.util.List.copyOf(columns);
            if (tables.isEmpty()) throw new IllegalArgumentException("SELECT requires at least one table");
        }
    }

    /**
     * INSERT intent.
     */
    record Insert(
        String rawQuery,
        double confidence,
        String table,
        java.util.List<String> columns,
        java.util.List<String> values
    ) implements SqlIntent {
        public Insert {
            columns = java.util.List.copyOf(columns);
            values = java.util.List.copyOf(values);
        }
    }

    /**
     * UPDATE intent.
     */
    record Update(
        String rawQuery,
        double confidence,
        String table,
        java.util.Map<String, String> assignments,
        String whereClause
    ) implements SqlIntent {
        public Update {
            assignments = java.util.Map.copyOf(assignments);
        }
    }

    /**
     * DELETE intent.
     */
    record Delete(
        String rawQuery,
        double confidence,
        String table,
        String whereClause
    ) implements SqlIntent {}

    /**
     * CREATE TABLE intent.
     */
    record CreateTable(
        String rawQuery,
        double confidence,
        String table,
        java.util.List<ColumnDef> columns
    ) implements SqlIntent {
        public CreateTable {
            columns = java.util.List.copyOf(columns);
        }
    }

    /**
     * ALTER TABLE intent.
     */
    record AlterTable(
        String rawQuery,
        double confidence,
        String table,
        String action,
        ColumnDef columnDef
    ) implements SqlIntent {}

    /**
     * DROP TABLE intent.
     */
    record DropTable(
        String rawQuery,
        double confidence,
        String table
    ) implements SqlIntent {}

    /**
     * JOIN query intent.
     */
    record Join(
        String rawQuery,
        double confidence,
        String leftTable,
        String rightTable,
        String joinType,
        String onClause
    ) implements SqlIntent {}

    /**
     * Aggregate query intent (COUNT, SUM, AVG, MIN, MAX).
     */
    record Aggregate(
        String rawQuery,
        double confidence,
        String function,
        String column,
        String table,
        String groupBy
    ) implements SqlIntent {}

    /**
     * Subquery intent.
     */
    record Subquery(
        String rawQuery,
        double confidence,
        SqlIntent outerIntent,
        SqlIntent innerIntent
    ) implements SqlIntent {}

    /**
     * EXPLAIN intent — user wants query plan explanation.
     */
    record Explain(
        String rawQuery,
        double confidence,
        String sqlToExplain
    ) implements SqlIntent {}

    /**
     * Schema information request intent.
     */
    record SchemaInfo(
        String rawQuery,
        double confidence,
        String table,
        String infoType  // "columns", "indexes", "constraints", "all"
    ) implements SqlIntent {}

    /**
     * Unknown / unresolvable intent.
     */
    record Unknown(
        String rawQuery,
        double confidence,
        String reason
    ) implements SqlIntent {}
}
