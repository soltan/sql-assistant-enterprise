package io.enterprise.sql.nlp;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Deterministic SQL-aware tokenizer.
 * No LLM, no ML model — pure rule-based tokenization.
 *
 * Tokenization strategy:
 * 1. Normalize whitespace and casing
 * 2. Split on word boundaries preserving SQL operators
 * 3. Classify each token (keyword, identifier, operator, literal, etc.)
 * 4. Produce token stream with position and type information
 *
 * Deterministic: same input always produces same token sequence.
 * O(n) time complexity where n = input length.
 */
public final class Tokenizer {

    /** SQL keywords set */
    private static final Set<String> SQL_KEYWORDS = Set.of(
        "select", "from", "where", "insert", "into", "values", "update", "set",
        "delete", "create", "table", "alter", "drop", "add", "column", "modify",
        "join", "inner", "left", "right", "outer", "full", "cross", "on",
        "and", "or", "not", "in", "like", "between", "is", "null", "exists",
        "group", "by", "order", "asc", "desc", "having", "limit", "offset",
        "union", "all", "intersect", "except", "distinct",
        "count", "sum", "avg", "min", "max",
        "as", "case", "when", "then", "else", "end",
        "true", "false", "primary", "key", "foreign", "references",
        "constraint", "unique", "check", "default", "index",
        "varchar", "integer", "int", "bigint", "smallint", "decimal", "numeric",
        "float", "double", "real", "boolean", "bool", "date", "time", "timestamp",
        "text", "char", "blob", "clob",
        "explain", "analyze", "show", "describe", "schema",
        "if", "not", "exists", "replace", "temporary", "temp"
    );

    /** SQL operators */
    private static final Pattern OPERATOR_PATTERN = Pattern.compile(
        "^(>=|<=|!=|<>|==|>|<|=|\\+|-|\\*|/|%|\\|\\||&&|\\.)$"
    );

    /** Token types */
    public enum TokenType {
        KEYWORD,        // SQL keyword (SELECT, FROM, WHERE, etc.)
        IDENTIFIER,     // Table or column name
        OPERATOR,       // SQL operator (=, >, <, +, -, etc.)
        STRING_LITERAL, // 'quoted string'
        NUMERIC_LITERAL,// 123, 45.67
        WILDCARD,       // *
        COMMA,          // ,
        SEMICOLON,      // ;
        LPAREN,         // (
        RPAREN,         // )
        DOT,            // .
        PLACEHOLDER,    // ? or :name
        UNKNOWN         // Unrecognized token
    }

    /**
     * Token record — immutable, Valhalla-ready.
     */
    public record Token(String text, TokenType type, int position) {}

    /**
     * Tokenize a SQL query or natural language input.
     */
    public static List<Token> tokenize(String input) {
        if (input == null || input.isBlank()) return List.of();

        List<Token> tokens = new ArrayList<>();
        int pos = 0;
        String trimmed = input.trim();

        while (pos < trimmed.length()) {
            char ch = trimmed.charAt(pos);

            // Skip whitespace
            if (Character.isWhitespace(ch)) {
                pos++;
                continue;
            }

            // String literal
            if (ch == '\'') {
                int start = pos;
                pos++; // skip opening quote
                while (pos < trimmed.length() && trimmed.charAt(pos) != '\'') {
                    pos++;
                }
                if (pos < trimmed.length()) pos++; // skip closing quote
                tokens.add(new Token(trimmed.substring(start, pos), TokenType.STRING_LITERAL, start));
                continue;
            }

            // Numeric literal
            if (Character.isDigit(ch) || (ch == '.' && pos + 1 < trimmed.length() &&
                Character.isDigit(trimmed.charAt(pos + 1)))) {
                int start = pos;
                boolean hasDot = false;
                while (pos < trimmed.length() &&
                    (Character.isDigit(trimmed.charAt(pos)) ||
                     (trimmed.charAt(pos) == '.' && !hasDot))) {
                    if (trimmed.charAt(pos) == '.') hasDot = true;
                    pos++;
                }
                tokens.add(new Token(trimmed.substring(start, pos), TokenType.NUMERIC_LITERAL, start));
                continue;
            }

            // Operators and multi-character operators
            if (pos + 1 < trimmed.length()) {
                String twoChar = trimmed.substring(pos, pos + 2);
                if (OPERATOR_PATTERN.matcher(twoChar).matches()) {
                    tokens.add(new Token(twoChar, TokenType.OPERATOR, pos));
                    pos += 2;
                    continue;
                }
            }
            if ("=><+-*/%".indexOf(ch) >= 0) {
                tokens.add(new Token(String.valueOf(ch), TokenType.OPERATOR, pos));
                pos++;
                continue;
            }

            // Punctuation
            if (ch == ',') { tokens.add(new Token(",", TokenType.COMMA, pos)); pos++; continue; }
            if (ch == ';') { tokens.add(new Token(";", TokenType.SEMICOLON, pos)); pos++; continue; }
            if (ch == '(') { tokens.add(new Token("(", TokenType.LPAREN, pos)); pos++; continue; }
            if (ch == ')') { tokens.add(new Token(")", TokenType.RPAREN, pos)); pos++; continue; }
            if (ch == '*') { tokens.add(new Token("*", TokenType.WILDCARD, pos)); pos++; continue; }
            if (ch == '?') { tokens.add(new Token("?", TokenType.PLACEHOLDER, pos)); pos++; continue; }
            if (ch == '.') { tokens.add(new Token(".", TokenType.DOT, pos)); pos++; continue; }

            // Placeholder (:name)
            if (ch == ':' && pos + 1 < trimmed.length() &&
                Character.isLetter(trimmed.charAt(pos + 1))) {
                int start = pos;
                pos++;
                while (pos < trimmed.length() &&
                    (Character.isLetterOrDigit(trimmed.charAt(pos)) ||
                     trimmed.charAt(pos) == '_')) {
                    pos++;
                }
                tokens.add(new Token(trimmed.substring(start, pos), TokenType.PLACEHOLDER, start));
                continue;
            }

            // Identifier or keyword
            if (Character.isLetter(ch) || ch == '_') {
                int start = pos;
                while (pos < trimmed.length() &&
                    (Character.isLetterOrDigit(trimmed.charAt(pos)) ||
                     trimmed.charAt(pos) == '_')) {
                    pos++;
                }
                String word = trimmed.substring(start, pos);
                String lowerWord = word.toLowerCase(Locale.ROOT);
                TokenType type = SQL_KEYWORDS.contains(lowerWord)
                    ? TokenType.KEYWORD
                    : TokenType.IDENTIFIER;
                tokens.add(new Token(word, type, start));
                continue;
            }

            // Unknown
            tokens.add(new Token(String.valueOf(ch), TokenType.UNKNOWN, pos));
            pos++;
        }

        return List.copyOf(tokens);
    }

    /**
     * Extract only the keyword tokens from a token list.
     */
    public static List<Token> keywords(List<Token> tokens) {
        return tokens.stream()
            .filter(t -> t.type() == TokenType.KEYWORD)
            .toList();
    }

    /**
     * Extract only the identifier tokens from a token list.
     */
    public static List<Token> identifiers(List<Token> tokens) {
        return tokens.stream()
            .filter(t -> t.type() == TokenType.IDENTIFIER)
            .toList();
    }

    /**
     * Check if a token list contains a specific keyword.
     */
    public static boolean containsKeyword(List<Token> tokens, String keyword) {
        return tokens.stream()
            .anyMatch(t -> t.type() == TokenType.KEYWORD &&
                t.text().equalsIgnoreCase(keyword));
    }
}
