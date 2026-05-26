package io.enterprise.sql.intent;

import io.enterprise.sql.model.IntentNode;
import io.enterprise.sql.model.SemanticHash;
import io.enterprise.sql.hashing.SimHash;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Intent resolution graph — the core deterministic routing engine.
 *
 * Structure:
 * - Root node: entry point for all queries
 * - Level 1: operation type (SELECT, INSERT, UPDATE, DELETE, DDL, META)
 * - Level 2: sub-type (JOIN, AGGREGATE, SUBQUERY for SELECT; etc.)
 * - Level 3: specific pattern variants
 *
 * Resolution strategy:
 * 1. Compute semantic hash of input query
 * 2. Traverse graph top-down, at each level finding the child
 *    with minimum Hamming distance to the query hash
 * 3. If distance exceeds threshold, fall back to broader pattern
 *
 * Properties:
 * - Deterministic: same query always resolves to same intent
 * - O(depth) resolution: typically 3-4 graph traversals
 * - Thread-safe: read-heavy, write-rarely (ConcurrentHashMap)
 */
public class IntentGraph {

    private static final int MAX_HAMMING_DISTANCE = 16;

    private final IntentNode root;
    private final Map<String, IntentNode> nodeIndex;
    private final Map<SemanticHash, IntentNode> hashIndex;

    private IntentGraph(IntentNode root, Map<String, IntentNode> nodeIndex,
                         Map<SemanticHash, IntentNode> hashIndex) {
        this.root = root;
        this.nodeIndex = nodeIndex;
        this.hashIndex = hashIndex;
    }

    /**
     * Resolve a query to its intent by traversing the graph.
     *
     * @param query natural language or SQL-like query
     * @return the best matching intent node
     */
    public ResolutionResult resolve(String query) {
        SemanticHash queryHash = SimHash.computeFromQuery(query);
        return resolveWithHash(query, queryHash);
    }

    /**
     * Resolve using a pre-computed semantic hash.
     */
    public ResolutionResult resolveWithHash(String query, SemanticHash queryHash) {
        List<IntentNode> path = new ArrayList<>();
        IntentNode current = root;
        path.add(current);

        // Traverse top-down
        while (!current.children().isEmpty()) {
            Optional<IntentNode> bestChild = current.findBestChild(queryHash, MAX_HAMMING_DISTANCE);
            if (bestChild.isEmpty()) break;
            current = bestChild.get();
            path.add(current);
        }

        double confidence = current.hash().similarity(queryHash);
        return new ResolutionResult(current, path, confidence, queryHash);
    }

    /**
     * Look up a node by its ID.
     */
    public Optional<IntentNode> getNode(String nodeId) {
        return Optional.ofNullable(nodeIndex.get(nodeId));
    }

    /**
     * Find nodes similar to a given hash.
     */
    public List<IntentNode> findSimilar(SemanticHash hash, int maxDistance) {
        return hashIndex.entrySet().stream()
            .filter(e -> e.getKey().isWithin(hash, maxDistance))
            .map(Map.Entry::getValue)
            .sorted(Comparator.comparingDouble(n -> n.hash().hammingDistance(hash)))
            .toList();
    }

    /**
     * Get the root node.
     */
    public IntentNode root() {
        return root;
    }

    /**
     * Resolution result record.
     */
    public record ResolutionResult(
        IntentNode matchedNode,
        List<IntentNode> resolutionPath,
        double confidence,
        SemanticHash queryHash
    ) {
        public ResolutionResult {
            resolutionPath = List.copyOf(resolutionPath);
        }

        /**
         * Depth of resolution in the graph.
         */
        public int depth() {
            return resolutionPath.size();
        }

        /**
         * Whether the resolution is high confidence.
         */
        public boolean isHighConfidence() {
            return confidence >= 0.8;
        }
    }

    /**
     * Builder for constructing the intent graph.
     */
    public static class Builder {
        private final Map<String, IntentNode> nodeIndex = new ConcurrentHashMap<>();
        private final Map<SemanticHash, IntentNode> hashIndex = new ConcurrentHashMap<>();

        /**
         * Build the default intent graph with pre-configured SQL intent patterns.
         */
        public IntentGraph buildDefault() {
            // Level 3: Leaf nodes with SQL templates
            IntentNode simpleSelect = IntentNode.leaf(
                "select_simple",
                SimHash.computeFromQuery("select columns from table"),
                "Simple SELECT", "Select",
                "SELECT {columns} FROM {table}", 10
            );

            IntentNode filteredSelect = IntentNode.leaf(
                "select_filtered",
                SimHash.computeFromQuery("select columns from table where condition"),
                "Filtered SELECT", "Select",
                "SELECT {columns} FROM {table} WHERE {condition}", 20
            );

            IntentNode joinSelect = IntentNode.leaf(
                "select_join",
                SimHash.computeFromQuery("select columns from table join table on condition"),
                "JOIN SELECT", "Join",
                "SELECT {columns} FROM {table1} {joinType} JOIN {table2} ON {condition}", 30
            );

            IntentNode aggregateSelect = IntentNode.leaf(
                "select_aggregate",
                SimHash.computeFromQuery("select count sum avg from table group by column"),
                "Aggregate SELECT", "Aggregate",
                "SELECT {function}({column}) FROM {table} GROUP BY {groupBy}", 25
            );

            IntentNode insertValues = IntentNode.leaf(
                "insert_values",
                SimHash.computeFromQuery("insert into table values data"),
                "INSERT values", "Insert",
                "INSERT INTO {table} ({columns}) VALUES ({values})", 10
            );

            IntentNode updateSet = IntentNode.leaf(
                "update_set",
                SimHash.computeFromQuery("update table set column value where condition"),
                "UPDATE set", "Update",
                "UPDATE {table} SET {assignments} WHERE {condition}", 10
            );

            IntentNode deleteWhere = IntentNode.leaf(
                "delete_where",
                SimHash.computeFromQuery("delete from table where condition"),
                "DELETE where", "Delete",
                "DELETE FROM {table} WHERE {condition}", 10
            );

            IntentNode createTable = IntentNode.leaf(
                "create_table",
                SimHash.computeFromQuery("create table with columns and types"),
                "CREATE TABLE", "CreateTable",
                "CREATE TABLE {table} ({columnDefs})", 10
            );

            IntentNode alterTable = IntentNode.leaf(
                "alter_table",
                SimHash.computeFromQuery("alter table add column modify"),
                "ALTER TABLE", "AlterTable",
                "ALTER TABLE {table} {action} {columnDef}", 10
            );

            IntentNode dropTable = IntentNode.leaf(
                "drop_table",
                SimHash.computeFromQuery("drop table remove"),
                "DROP TABLE", "DropTable",
                "DROP TABLE {table}", 10
            );

            IntentNode explainQuery = IntentNode.leaf(
                "explain_query",
                SimHash.computeFromQuery("explain query plan execution"),
                "EXPLAIN", "Explain",
                "EXPLAIN {sql}", 10
            );

            IntentNode schemaInfo = IntentNode.leaf(
                "schema_info",
                SimHash.computeFromQuery("show schema table columns indexes constraints"),
                "Schema Info", "SchemaInfo",
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'", 10
            );

            // Level 2: Sub-type groups
            IntentNode selectGroup = new IntentNode(
                "select_group",
                SimHash.computeFromQuery("select query get data retrieve"),
                "SELECT group", "Select",
                "", List.of(simpleSelect, filteredSelect, joinSelect, aggregateSelect), 50
            );

            IntentNode insertGroup = new IntentNode(
                "insert_group",
                SimHash.computeFromQuery("insert add new data create row"),
                "INSERT group", "Insert",
                "", List.of(insertValues), 40
            );

            IntentNode updateGroup = new IntentNode(
                "update_group",
                SimHash.computeFromQuery("update modify change existing data"),
                "UPDATE group", "Update",
                "", List.of(updateSet), 40
            );

            IntentNode deleteGroup = new IntentNode(
                "delete_group",
                SimHash.computeFromQuery("delete remove row data"),
                "DELETE group", "Delete",
                "", List.of(deleteWhere), 40
            );

            IntentNode ddlGroup = new IntentNode(
                "ddl_group",
                SimHash.computeFromQuery("create alter drop table schema structure"),
                "DDL group", "CreateTable",
                "", List.of(createTable, alterTable, dropTable), 30
            );

            IntentNode metaGroup = new IntentNode(
                "meta_group",
                SimHash.computeFromQuery("explain schema info describe structure"),
                "META group", "Explain",
                "", List.of(explainQuery, schemaInfo), 20
            );

            // Level 1: Root
            IntentNode rootNode = new IntentNode(
                "root",
                SemanticHash.ZERO,
                "Root", "Unknown",
                "", List.of(selectGroup, insertGroup, updateGroup, deleteGroup, ddlGroup, metaGroup), 0
            );

            // Build indexes
            indexNode(rootNode);
            indexNode(selectGroup); indexNode(insertGroup); indexNode(updateGroup);
            indexNode(deleteGroup); indexNode(ddlGroup); indexNode(metaGroup);
            indexNode(simpleSelect); indexNode(filteredSelect); indexNode(joinSelect);
            indexNode(aggregateSelect); indexNode(insertValues); indexNode(updateSet);
            indexNode(deleteWhere); indexNode(createTable); indexNode(alterTable);
            indexNode(dropTable); indexNode(explainQuery); indexNode(schemaInfo);

            return new IntentGraph(rootNode, nodeIndex, hashIndex);
        }

        private void indexNode(IntentNode node) {
            nodeIndex.put(node.id(), node);
            hashIndex.put(node.hash(), node);
        }
    }
}
