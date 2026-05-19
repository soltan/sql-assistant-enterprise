package io.enterprise.sql.model;

import java.util.Map;

/**
 * Represents a fully resolved SQL query ready for execution.
 * Immutable value object — Valhalla-ready (no identity semantics needed).
 *
 * @param sql        parameterized SQL string
 * @param parameters bind parameters (name → value)
 * @param intent     the resolved SqlIntent that generated this query
 * @param estimatedCost  estimated execution cost (0 = unknown)
 */
public record SqlQuery(
    String sql,
    Map<String, Object> parameters,
    SqlIntent intent,
    double estimatedCost
) {
    public SqlQuery {
        parameters = Map.copyOf(parameters);
    }

    /**
     * Create a simple query with no parameters.
     */
    public static SqlQuery of(String sql, SqlIntent intent) {
        return new SqlQuery(sql, Map.of(), intent, 0.0);
    }

    /**
     * Create a parameterized query.
     */
    public static SqlQuery parameterized(String sql, Map<String, Object> params, SqlIntent intent) {
        return new SqlQuery(sql, params, intent, 0.0);
    }

    /**
     * Create a query with cost estimate.
     */
    public static SqlQuery withCost(String sql, Map<String, Object> params,
                                     SqlIntent intent, double cost) {
        return new SqlQuery(sql, params, intent, cost);
    }

    /**
     * Render the SQL with inline parameters (for debugging / logging).
     */
    public String toInlineSql() {
        String result = sql;
        for (var entry : parameters.entrySet()) {
            String placeholder = ":" + entry.getKey();
            String value = entry.getValue() instanceof String s
                ? "'" + s + "'"
                : String.valueOf(entry.getValue());
            result = result.replace(placeholder, value);
        }
        return result;
    }

    @Override
    public String toString() {
        return "SqlQuery{sql='%s', params=%s, intent=%s, cost=%.2f}"
            .formatted(sql, parameters.keySet(), intent.getClass().getSimpleName(), estimatedCost);
    }
}
