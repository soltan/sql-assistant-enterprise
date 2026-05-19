package io.enterprise.sql.model;

import java.util.List;
import java.util.Optional;

/**
 * A node in the intent resolution graph.
 * Each node represents a known intent pattern with its semantic hash,
 * associated SQL template, and edges to sub-intent nodes.
 *
 * Data Oriented: immutable, thread-safe, no mutation.
 * Pattern: each node is a self-contained resolution unit.
 *
 * @param id           unique node identifier
 * @param hash         semantic hash of the intent pattern
 * @param label        human-readable label
 * @param intentType   the SqlIntent subtype this node resolves to
 * @param sqlTemplate  parameterized SQL template (e.g., "SELECT {columns} FROM {table} WHERE {condition}")
 * @param children     sub-intent nodes for multi-step resolution
 * @param priority     resolution priority (higher = preferred when multiple matches)
 */
public record IntentNode(
    String id,
    SemanticHash hash,
    String label,
    String intentType,
    String sqlTemplate,
    List<IntentNode> children,
    int priority
) {
    public IntentNode {
        children = List.copyOf(children);
    }

    /**
     * Find a child node by label.
     */
    public Optional<IntentNode> findChild(String childLabel) {
        return children.stream()
            .filter(c -> c.label().equals(childLabel))
            .findFirst();
    }

    /**
     * Find the child with highest priority within Hamming distance.
     */
    public Optional<IntentNode> findBestChild(SemanticHash query, int maxDistance) {
        return children.stream()
            .filter(c -> c.hash().isWithin(query, maxDistance))
            .max(java.util.Comparator.comparingInt(IntentNode::priority));
    }

    /**
     * Convenience factory for leaf nodes (no children).
     */
    public static IntentNode leaf(String id, SemanticHash hash, String label,
                                   String intentType, String sqlTemplate, int priority) {
        return new IntentNode(id, hash, label, intentType, sqlTemplate, List.of(), priority);
    }
}
