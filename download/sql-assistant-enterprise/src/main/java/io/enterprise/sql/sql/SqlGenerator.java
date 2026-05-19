package io.enterprise.sql.sql;

import io.enterprise.sql.model.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Deterministic SQL generator — converts SqlIntent into parameterized SQL.
 * No LLM, no template engine — pure pattern matching on sealed type hierarchy.
 *
 * Uses exhaustive pattern matching on the SqlIntent sealed interface
 * to guarantee all intent types are handled at compile time.
 */
public class SqlGenerator {

    /**
     * Generate SQL from a resolved intent.
     * Exhaustive switch on sealed SqlIntent — compiler enforces completeness.
     */
    public SqlQuery generate(SqlIntent intent) {
        return switch (intent) {
            case SqlIntent.Select s      -> generateSelect(s);
            case SqlIntent.Insert i      -> generateInsert(i);
            case SqlIntent.Update u      -> generateUpdate(u);
            case SqlIntent.Delete d      -> generateDelete(d);
            case SqlIntent.CreateTable c -> generateCreateTable(c);
            case SqlIntent.AlterTable a  -> generateAlterTable(a);
            case SqlIntent.DropTable dt  -> generateDropTable(dt);
            case SqlIntent.Join j        -> generateJoin(j);
            case SqlIntent.Aggregate ag  -> generateAggregate(ag);
            case SqlIntent.Subquery sq   -> generateSubquery(sq);
            case SqlIntent.Explain ex    -> generateExplain(ex);
            case SqlIntent.SchemaInfo si -> generateSchemaInfo(si);
            case SqlIntent.Unknown uk    -> generateUnknown(uk);
        };
    }

    private SqlQuery generateSelect(SqlIntent.Select s) {
        StringBuilder sql = new StringBuilder("SELECT ");
        Map<String, Object> params = new LinkedHashMap<>();

        // Columns
        if (s.columns().isEmpty() || s.columns().contains("*")) {
            sql.append("*");
        } else {
            sql.append(String.join(", ", s.columns()));
        }

        // FROM
        sql.append(" FROM ").append(String.join(", ", s.tables()));

        // WHERE
        if (s.whereClause() != null && !s.whereClause().isEmpty()) {
            sql.append(" WHERE :where_condition");
            params.put("where_condition", s.whereClause());
        }

        // ORDER BY
        if (s.orderBy() != null && !s.orderBy().isEmpty()) {
            sql.append(" ORDER BY :order_by");
            params.put("order_by", s.orderBy());
        }

        // LIMIT
        if (s.limit() > 0) {
            sql.append(" LIMIT :limit");
            params.put("limit", s.limit());
        }

        return SqlQuery.parameterized(sql.toString(), params, s);
    }

    private SqlQuery generateInsert(SqlIntent.Insert i) {
        Map<String, Object> params = new LinkedHashMap<>();
        String cols = i.columns().isEmpty() ? "" :
            " (" + String.join(", ", i.columns()) + ")";
        String placeholders = i.values().isEmpty() ? "" :
            i.values().stream().map(v -> "?").collect(Collectors.joining(", "));

        String sql = "INSERT INTO " + i.table() + cols + " VALUES (" + placeholders + ")";

        for (int idx = 0; idx < i.values().size(); idx++) {
            params.put("value_" + idx, i.values().get(idx));
        }

        return SqlQuery.parameterized(sql, params, i);
    }

    private SqlQuery generateUpdate(SqlIntent.Update u) {
        Map<String, Object> params = new LinkedHashMap<>();
        StringBuilder sql = new StringBuilder("UPDATE ").append(u.table()).append(" SET ");

        String setClause = u.assignments().entrySet().stream()
            .map(e -> e.getKey() + " = :set_" + e.getKey())
            .collect(Collectors.joining(", "));
        sql.append(setClause);

        u.assignments().forEach((k, v) -> params.put("set_" + k, v));

        if (u.whereClause() != null && !u.whereClause().isEmpty()) {
            sql.append(" WHERE :where_condition");
            params.put("where_condition", u.whereClause());
        }

        return SqlQuery.parameterized(sql.toString(), params, u);
    }

    private SqlQuery generateDelete(SqlIntent.Delete d) {
        Map<String, Object> params = new LinkedHashMap<>();
        StringBuilder sql = new StringBuilder("DELETE FROM ").append(d.table());

        if (d.whereClause() != null && !d.whereClause().isEmpty()) {
            sql.append(" WHERE :where_condition");
            params.put("where_condition", d.whereClause());
        }

        return SqlQuery.parameterized(sql.toString(), params, d);
    }

    private SqlQuery generateCreateTable(SqlIntent.CreateTable c) {
        String columnDefs = c.columns().stream()
            .map(col -> {
                StringBuilder def = new StringBuilder(col.name()).append(" ").append(col.dataType());
                if (col.primaryKey()) def.append(" PRIMARY KEY");
                if (!col.nullable()) def.append(" NOT NULL");
                return def.toString();
            })
            .collect(Collectors.joining(", "));

        String sql = "CREATE TABLE " + c.table() + " (" + columnDefs + ")";
        return SqlQuery.of(sql, c);
    }

    private SqlQuery generateAlterTable(SqlIntent.AlterTable a) {
        StringBuilder sql = new StringBuilder("ALTER TABLE ").append(a.table());
        sql.append(" ").append(a.action()).append(" COLUMN ");

        if (a.columnDef() != null) {
            ColumnDef col = a.columnDef();
            sql.append(col.name()).append(" ").append(col.dataType());
            if (!col.nullable()) sql.append(" NOT NULL");
        }

        return SqlQuery.of(sql.toString(), a);
    }

    private SqlQuery generateDropTable(SqlIntent.DropTable dt) {
        return SqlQuery.of("DROP TABLE " + dt.table(), dt);
    }

    private SqlQuery generateJoin(SqlIntent.Join j) {
        String sql = "SELECT * FROM %s %s JOIN %s ON %s".formatted(
            j.leftTable(),
            j.joinType(),
            j.rightTable(),
            j.onClause().isEmpty() ? ":join_condition" : j.onClause()
        );
        Map<String, Object> params = new LinkedHashMap<>();
        if (j.onClause().isEmpty()) {
            params.put("join_condition", "1=1");
        }
        return SqlQuery.parameterized(sql, params, j);
    }

    private SqlQuery generateAggregate(SqlIntent.Aggregate ag) {
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT ").append(ag.function()).append("(").append(ag.column()).append(")");
        sql.append(" FROM ").append(ag.table());

        Map<String, Object> params = new LinkedHashMap<>();

        if (ag.groupBy() != null && !ag.groupBy().isEmpty()) {
            sql.append(" GROUP BY :group_by");
            params.put("group_by", ag.groupBy());
        }

        return SqlQuery.parameterized(sql.toString(), params, ag);
    }

    private SqlQuery generateSubquery(SqlIntent.Subquery sq) {
        SqlQuery outer = generate(sq.outerIntent());
        SqlQuery inner = generate(sq.innerIntent());
        // Replace inner query placeholder with subquery
        String combined = outer.sql().replace(":subquery", "(" + inner.sql() + ")");
        Map<String, Object> combinedParams = new LinkedHashMap<>(outer.parameters());
        combinedParams.putAll(inner.parameters());
        return SqlQuery.parameterized(combined, combinedParams, sq);
    }

    private SqlQuery generateExplain(SqlIntent.Explain ex) {
        String sql = "EXPLAIN " + ex.sqlToExplain();
        return SqlQuery.of(sql, ex);
    }

    private SqlQuery generateSchemaInfo(SqlIntent.SchemaInfo si) {
        String sql = switch (si.infoType()) {
            case "columns" ->
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = ':table'";
            case "indexes" ->
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = ':table'";
            case "constraints" ->
                "SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name = ':table'";
            default ->
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = ':table'";
        };
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("table", si.table());
        return SqlQuery.parameterized(sql, params, si);
    }

    private SqlQuery generateUnknown(SqlIntent.Unknown uk) {
        return SqlQuery.of("-- Unable to generate SQL: " + uk.reason(), uk);
    }
}
