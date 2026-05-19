package io.enterprise.sql.sql;

import io.enterprise.sql.model.SqlIntent;
import io.enterprise.sql.model.SqlQuery;

import java.util.*;
import java.util.regex.Pattern;

/**
 * SQL query validator — deterministic, rule-based validation.
 * No LLM, no external dependency — pure pattern matching and syntax rules.
 *
 * Validation checks:
 * 1. SQL injection patterns
 * 2. Syntax structure validation
 * 3. Parameter completeness
 * 4. Dangerous operation detection
 * 5. Schema consistency (if schema resolver available)
 */
public class QueryValidator {

    /** SQL injection patterns */
    private static final List<Pattern> INJECTION_PATTERNS = List.of(
        Pattern.compile("(?i)(--|/\\*|\\*/|xp_|sp_)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(union\\s+select)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(;\\s*(drop|alter|truncate|delete|update))", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(1\\s*=\\s*1)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(or\\s+1\\s*=\\s*1)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(waitfor\\s+delay)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(benchmark\\s*\\()", Pattern.CASE_INSENSITIVE),
        Pattern.compile("(?i)(sleep\\s*\\()", Pattern.CASE_INSENSITIVE)
    );

    /** Dangerous operations that require explicit confirmation */
    private static final Set<String> DANGEROUS_OPERATIONS = Set.of(
        "DROP", "TRUNCATE", "DELETE WITHOUT WHERE", "UPDATE WITHOUT WHERE"
    );

    private final SchemaResolver schemaResolver;
    private final boolean strictMode;

    public QueryValidator(SchemaResolver schemaResolver, boolean strictMode) {
        this.schemaResolver = schemaResolver;
        this.strictMode = strictMode;
    }

    public QueryValidator() {
        this(null, true);
    }

    /**
     * Validate a generated SQL query.
     *
     * @param query the SQL query to validate
     * @return validation result with status and messages
     */
    public ValidationResult validate(SqlQuery query) {
        List<String> warnings = new ArrayList<>();
        List<String> errors = new ArrayList<>();

        String sql = query.sql();

        // 1. Empty check
        if (sql == null || sql.isBlank()) {
            errors.add("Generated SQL is empty");
            return new ValidationResult(false, errors, warnings);
        }

        // 2. SQL injection check
        for (Pattern pattern : INJECTION_PATTERNS) {
            if (pattern.matcher(sql).find()) {
                errors.add("Potential SQL injection detected: pattern " + pattern.pattern());
            }
        }

        // 3. Multiple statement check
        long semicolonCount = sql.chars().filter(c -> c == ';').count();
        if (semicolonCount > 1) {
            errors.add("Multiple SQL statements detected (potential injection)");
        }

        // 4. Parameter completeness check
        long placeholderCount = countPlaceholders(sql);
        long paramCount = query.parameters().size();
        if (placeholderCount != paramCount && paramCount > 0) {
            warnings.add("Placeholder count (%d) doesn't match parameter count (%d)"
                .formatted(placeholderCount, paramCount));
        }

        // 5. Dangerous operation detection
        checkDangerousOperations(sql, warnings);

        // 6. Schema consistency check (if resolver available)
        if (schemaResolver != null) {
            checkSchemaConsistency(query, warnings);
        }

        // 7. Query length check
        if (sql.length() > 65536) {
            errors.add("SQL query exceeds maximum length (64KB)");
        }

        // 8. Unterminated string check
        if (countChar(sql, '\'') % 2 != 0) {
            errors.add("Unterminated string literal detected");
        }

        boolean valid = errors.isEmpty();
        return new ValidationResult(valid, errors, warnings);
    }

    /**
     * Check for dangerous operations.
     */
    private void checkDangerousOperations(String sql, List<String> warnings) {
        String upper = sql.toUpperCase(Locale.ROOT).trim();

        if (upper.startsWith("DROP")) {
            warnings.add("DANGEROUS: DROP operation detected");
        }

        if (upper.startsWith("TRUNCATE")) {
            warnings.add("DANGEROUS: TRUNCATE operation detected");
        }

        if (upper.startsWith("DELETE") && !upper.contains("WHERE")) {
            warnings.add("DANGEROUS: DELETE without WHERE clause will affect all rows");
        }

        if (upper.startsWith("UPDATE") && !upper.contains("WHERE")) {
            warnings.add("DANGEROUS: UPDATE without WHERE clause will affect all rows");
        }
    }

    /**
     * Check schema consistency if resolver is available.
     */
    private void checkSchemaConsistency(SqlQuery query, List<String> warnings) {
        // Check if referenced tables exist in schema
        if (query.intent() instanceof SqlIntent.Select s) {
            for (String table : s.tables()) {
                if (schemaResolver.resolveTableName(table).isEmpty()) {
                    warnings.add("Table '%s' not found in schema".formatted(table));
                }
            }
        }
    }

    /**
     * Count named placeholders in SQL (:name format).
     */
    private long countPlaceholders(String sql) {
        return Arrays.stream(sql.split("\\s+"))
            .filter(t -> t.startsWith(":") && t.length() > 1)
            .count();
    }

    /**
     * Count occurrences of a character in a string.
     */
    private long countChar(String s, char c) {
        return s.chars().filter(ch -> ch == c).count();
    }

    /**
     * Validation result record.
     */
    public record ValidationResult(
        boolean valid,
        List<String> errors,
        List<String> warnings
    ) {
        public ValidationResult {
            errors = List.copyOf(errors);
            warnings = List.copyOf(warnings);
        }

        /**
         * Combine with another validation result.
         */
        public ValidationResult and(ValidationResult other) {
            var combinedErrors = new ArrayList<>(errors);
            combinedErrors.addAll(other.errors);
            var combinedWarnings = new ArrayList<>(warnings);
            combinedWarnings.addAll(other.warnings);
            return new ValidationResult(valid && other.valid, combinedErrors, combinedWarnings);
        }

        /**
         * Format as a human-readable string.
         */
        public String format() {
            StringBuilder sb = new StringBuilder();
            if (valid) {
                sb.append("VALID");
            } else {
                sb.append("INVALID");
            }
            if (!errors.isEmpty()) {
                sb.append("\n  Errors: ").append(String.join("; ", errors));
            }
            if (!warnings.isEmpty()) {
                sb.append("\n  Warnings: ").append(String.join("; ", warnings));
            }
            return sb.toString();
        }
    }
}
