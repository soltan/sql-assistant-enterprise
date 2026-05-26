package io.enterprise.sql;

import io.enterprise.sql.config.AssistantConfig;
import io.enterprise.sql.hashing.SimHash;
import io.enterprise.sql.hashing.SemanticHasher;
import io.enterprise.sql.intent.IntentGraph;
import io.enterprise.sql.intent.IntentResolver;
import io.enterprise.sql.jakarta.JakartaAIAdapter;
import io.enterprise.sql.jakarta.SqlAgent;
import io.enterprise.sql.model.*;
import io.enterprise.sql.nlp.EmbeddingEngine;
import io.enterprise.sql.nlp.Tokenizer;
import io.enterprise.sql.sql.QueryValidator;
import io.enterprise.sql.sql.SchemaResolver;
import io.enterprise.sql.sql.SqlGenerator;
import io.enterprise.sql.vector.HNSWIndex;
import io.enterprise.sql.vector.VectorOps;
import org.junit.jupiter.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the Enterprise SQL Assistant.
 */
class SqlAssistantTest {

    private SqlAgent agent;
    private AssistantConfig config;

    @BeforeEach
    void setUp() {
        config = AssistantConfig.development();
        agent = new SqlAgent(config);
    }

    // ==================== DOP Model Tests ====================

    @Test
    void testSemanticHashHammingDistance() {
        SemanticHash h1 = new SemanticHash(0xFFFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL);
        SemanticHash h2 = new SemanticHash(0x0000000000000000L, 0x0000000000000000L);
        assertEquals(128, h1.hammingDistance(h2));

        SemanticHash h3 = new SemanticHash(0xFFFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL);
        assertEquals(0, h1.hammingDistance(h3));
    }

    @Test
    void testSemanticHashSimilarity() {
        SemanticHash h1 = new SemanticHash(0xFFFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL);
        SemanticHash h2 = new SemanticHash(0x0000000000000000L, 0x0000000000000000L);
        assertEquals(0.0, h1.similarity(h2), 0.001);

        SemanticHash h3 = new SemanticHash(0xFFFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL);
        assertEquals(1.0, h1.similarity(h3), 0.001);
    }

    @Test
    void testSemanticHashIsWithin() {
        SemanticHash h1 = new SemanticHash(0b1111111111111111L, 0L);
        SemanticHash h2 = new SemanticHash(0b1111111111111110L, 0L);
        assertTrue(h1.isWithin(h2, 1));
        assertFalse(h1.isWithin(h2, 0));
    }

    @Test
    void testSqlIntentSealedHierarchy() {
        SqlIntent select = new SqlIntent.Select(
            "select * from users", 0.95,
            List.of("users"), List.of(), "", "", 0
        );
        assertInstanceOf(SqlIntent.Select.class, select);
        assertEquals("users", ((SqlIntent.Select) select).tables().getFirst());
    }

    @Test
    void testSqlIntentSelectValidation() {
        assertThrows(IllegalArgumentException.class, () ->
            new SqlIntent.Select("bad query", 0.5, List.of(), List.of(), "", "", 0)
        );
    }

    @Test
    void testSqlQueryParameterized() {
        SqlIntent intent = new SqlIntent.Select(
            "select * from users where active = true", 0.9,
            List.of("users"), List.of(), "active = true", "", 0
        );
        SqlQuery query = SqlQuery.parameterized(
            "SELECT * FROM users WHERE active = :active",
            Map.of("active", true), intent
        );
        assertEquals("SELECT * FROM users WHERE active = :active", query.sql());
        assertTrue(query.parameters().containsKey("active"));
    }

    // ==================== SimHash Tests ====================

    @Test
    void testSimHashDeterministic() {
        SemanticHash h1 = SimHash.computeFromQuery("select * from users");
        SemanticHash h2 = SimHash.computeFromQuery("select * from users");
        assertEquals(h1, h2);
    }

    @Test
    void testSimHashSimilarity() {
        SemanticHash h1 = SimHash.computeFromQuery("select * from users");
        SemanticHash h2 = SimHash.computeFromQuery("select all from users");
        SemanticHash h3 = SimHash.computeFromQuery("delete from orders where id = 1");

        double simSame = h1.similarity(h2);
        double simDiff = h1.similarity(h3);

        assertTrue(simSame > simDiff,
            "Similar queries should have higher similarity than different queries");
    }

    @Test
    void testSimHashEmpty() {
        SemanticHash hash = SimHash.computeFromQuery("");
        assertEquals(SemanticHash.ZERO, hash);
    }

    // ==================== Intent Graph Tests ====================

    @Test
    void testIntentGraphResolution() {
        IntentGraph graph = new IntentGraph.Builder().buildDefault();
        IntentGraph.ResolutionResult result = graph.resolve("select columns from table where condition");
        assertTrue(result.confidence() > 0.0);
        assertFalse(result.resolutionPath().isEmpty());
    }

    @Test
    void testIntentGraphSelectQuery() {
        IntentGraph graph = new IntentGraph.Builder().buildDefault();
        IntentGraph.ResolutionResult result = graph.resolve("select * from users where active = true");
        assertEquals("Select", result.matchedNode().intentType());
    }

    @Test
    void testIntentGraphInsertQuery() {
        IntentGraph graph = new IntentGraph.Builder().buildDefault();
        IntentGraph.ResolutionResult result = graph.resolve("insert into table values data");
        assertNotNull(result.matchedNode());
    }

    // ==================== Intent Resolver Tests ====================

    @Test
    void testIntentResolverSelect() {
        SemanticHasher hasher = new SemanticHasher();
        IntentGraph graph = new IntentGraph.Builder().buildDefault();
        IntentResolver resolver = new IntentResolver(graph, hasher);

        SqlIntent intent = resolver.resolve("select name, email from users where active = true");
        assertInstanceOf(SqlIntent.Select.class, intent);
    }

    @Test
    void testIntentResolverInsert() {
        SemanticHasher hasher = new SemanticHasher();
        IntentGraph graph = new IntentGraph.Builder().buildDefault();
        IntentResolver resolver = new IntentResolver(graph, hasher);

        SqlIntent intent = resolver.resolve("insert into users values (1, 'test')");
        assertInstanceOf(SqlIntent.Insert.class, intent);
    }

    @Test
    void testIntentResolverDelete() {
        SemanticHasher hasher = new SemanticHasher();
        IntentGraph graph = new IntentGraph.Builder().buildDefault();
        IntentResolver resolver = new IntentResolver(graph, hasher);

        SqlIntent intent = resolver.resolve("delete from users where id = 5");
        assertInstanceOf(SqlIntent.Delete.class, intent);
    }

    // ==================== SQL Generator Tests ====================

    @Test
    void testSqlGeneratorSelect() {
        SqlGenerator generator = new SqlGenerator();
        SqlIntent select = new SqlIntent.Select(
            "select * from users", 0.95,
            List.of("users"), List.of(), "", "", 0
        );
        SqlQuery query = generator.generate(select);
        assertTrue(query.sql().startsWith("SELECT"));
        assertTrue(query.sql().contains("FROM users"));
    }

    @Test
    void testSqlGeneratorSelectWithWhere() {
        SqlGenerator generator = new SqlGenerator();
        SqlIntent select = new SqlIntent.Select(
            "select * from users where active = true", 0.95,
            List.of("users"), List.of(), "active = true", "", 0
        );
        SqlQuery query = generator.generate(select);
        assertTrue(query.sql().contains("WHERE"));
    }

    @Test
    void testSqlGeneratorInsert() {
        SqlGenerator generator = new SqlGenerator();
        SqlIntent insert = new SqlIntent.Insert(
            "insert into users values", 0.9,
            "users", List.of("name", "email"), List.of("'John'", "'john@test.com'")
        );
        SqlQuery query = generator.generate(insert);
        assertTrue(query.sql().startsWith("INSERT INTO users"));
    }

    @Test
    void testSqlGeneratorCreateTable() {
        SqlGenerator generator = new SqlGenerator();
        SqlIntent create = new SqlIntent.CreateTable(
            "create table users", 0.9,
            "users", List.of(
                ColumnDef.primaryKey("id", "BIGINT"),
                ColumnDef.of("name", "VARCHAR(255)")
            )
        );
        SqlQuery query = generator.generate(create);
        assertTrue(query.sql().startsWith("CREATE TABLE users"));
        assertTrue(query.sql().contains("PRIMARY KEY"));
        assertTrue(query.sql().contains("NOT NULL"));
    }

    @Test
    void testSqlGeneratorExhaustiveSwitch() {
        SqlGenerator generator = new SqlGenerator();
        // Test all intent types compile without error
        List<SqlIntent> intents = List.of(
            new SqlIntent.Select("q", 0.9, List.of("t"), List.of(), "", "", 0),
            new SqlIntent.Insert("q", 0.9, "t", List.of(), List.of()),
            new SqlIntent.Update("q", 0.9, "t", Map.of(), ""),
            new SqlIntent.Delete("q", 0.9, "t", ""),
            new SqlIntent.CreateTable("q", 0.9, "t", List.of()),
            new SqlIntent.AlterTable("q", 0.9, "t", "ADD", null),
            new SqlIntent.DropTable("q", 0.9, "t"),
            new SqlIntent.Join("q", 0.9, "a", "b", "INNER", "a.id = b.id"),
            new SqlIntent.Aggregate("q", 0.9, "COUNT", "*", "t", ""),
            new SqlIntent.Explain("q", 0.9, "SELECT * FROM t"),
            new SqlIntent.SchemaInfo("q", 0.9, "t", "columns"),
            new SqlIntent.Unknown("q", 0.1, "test")
        );

        for (SqlIntent intent : intents) {
            SqlQuery query = generator.generate(intent);
            assertNotNull(query);
            assertNotNull(query.sql());
        }
    }

    // ==================== Tokenizer Tests ====================

    @Test
    void testTokenizerBasic() {
        List<Tokenizer.Token> tokens = Tokenizer.tokenize("SELECT * FROM users WHERE id = 1");
        assertFalse(tokens.isEmpty());
        assertTrue(tokens.stream().anyMatch(t -> t.type() == Tokenizer.TokenType.KEYWORD));
    }

    @Test
    void testTokenizerStringLiteral() {
        List<Tokenizer.Token> tokens = Tokenizer.tokenize("SELECT * FROM users WHERE name = 'John'");
        assertTrue(tokens.stream().anyMatch(t -> t.type() == Tokenizer.TokenType.STRING_LITERAL));
    }

    @Test
    void testTokenizerNumericLiteral() {
        List<Tokenizer.Token> tokens = Tokenizer.tokenize("SELECT * FROM users WHERE id = 42");
        assertTrue(tokens.stream().anyMatch(t -> t.type() == Tokenizer.TokenType.NUMERIC_LITERAL));
    }

    // ==================== Vector Operations Tests ====================

    @Test
    void testVectorOpsCosineSimilarity() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {0.0f, 1.0f, 0.0f};
        float sim = VectorOps.cosineSimilarity(a, b);
        assertEquals(0.0f, sim, 0.001f);
    }

    @Test
    void testVectorOpsIdenticalVectors() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float sim = VectorOps.cosineSimilarity(a, a);
        assertEquals(1.0f, sim, 0.001f);
    }

    @Test
    void testVectorOpsDotProduct() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {4.0f, 5.0f, 6.0f};
        float dot = VectorOps.dotProduct(a, b);
        assertEquals(32.0f, dot, 0.001f); // 1*4 + 2*5 + 3*6 = 32
    }

    // ==================== HNSW Index Tests ====================

    @Test
    void testHNSWInsertAndSearch() {
        HNSWIndex index = new HNSWIndex(4, 8, 50);

        SemanticHash h1 = new SemanticHash(1L, 0L);
        SemanticHash h2 = new SemanticHash(2L, 0L);
        SemanticHash h3 = new SemanticHash(3L, 0L);

        index.insert(1, new float[]{1.0f, 0.0f, 0.0f, 0.0f}, h1);
        index.insert(2, new float[]{0.0f, 1.0f, 0.0f, 0.0f}, h2);
        index.insert(3, new float[]{0.9f, 0.1f, 0.0f, 0.0f}, h3);

        var results = index.search(new float[]{1.0f, 0.0f, 0.0f, 0.0f}, 2, 10);
        assertFalse(results.isEmpty());
        assertEquals(1L, results.getFirst().id()); // Closest to (1,0,0,0) should be ID 1
    }

    // ==================== Embedding Engine Tests ====================

    @Test
    void testEmbeddingDeterministic() {
        EmbeddingEngine engine = new EmbeddingEngine(64);
        float[] e1 = engine.embed("select * from users");
        float[] e2 = engine.embed("select * from users");
        assertArrayEquals(e1, e2);
    }

    @Test
    void testEmbeddingDimension() {
        EmbeddingEngine engine = new EmbeddingEngine(128);
        float[] embedding = engine.embed("test query");
        assertEquals(128, embedding.length);
    }

    @Test
    void testEmbeddingSimilarity() {
        EmbeddingEngine engine = new EmbeddingEngine(128);
        float sim1 = engine.similarity("select * from users", "select all from users");
        float sim2 = engine.similarity("select * from users", "delete from orders");
        assertTrue(sim1 > sim2, "Similar queries should have higher embedding similarity");
    }

    // ==================== Query Validator Tests ====================

    @Test
    void testQueryValidatorValidQuery() {
        QueryValidator validator = new QueryValidator();
        SqlQuery query = SqlQuery.of("SELECT * FROM users WHERE active = true",
            new SqlIntent.Select("q", 0.9, List.of("users"), List.of(), "", "", 0));
        var result = validator.validate(query);
        assertTrue(result.valid());
    }

    @Test
    void testQueryValidatorInjectionDetection() {
        QueryValidator validator = new QueryValidator();
        SqlQuery query = SqlQuery.of("SELECT * FROM users; DROP TABLE users",
            new SqlIntent.Unknown("q", 0.1, "injection attempt"));
        var result = validator.validate(query);
        assertFalse(result.warnings().isEmpty() && result.errors().isEmpty());
    }

    // ==================== Schema Resolver Tests ====================

    @Test
    void testSchemaResolverDefaults() {
        SchemaResolver resolver = new SchemaResolver();
        resolver.loadDefaults();
        assertTrue(resolver.getTable("users").isPresent());
        assertTrue(resolver.getTable("orders").isPresent());
        assertEquals(3, resolver.tableNames().size());
    }

    @Test
    void testSchemaResolverResolveTableName() {
        SchemaResolver resolver = new SchemaResolver();
        resolver.loadDefaults();
        assertEquals("users", resolver.resolveTableName("users").orElse(""));
        assertEquals("users", resolver.resolveTableName("Users").orElse(""));
    }

    // ==================== ScopedValue Config Tests ====================

    @Test
    void testScopedValueConfigBinding() {
        AssistantConfig config = AssistantConfig.defaults();
        assertFalse(AssistantConfig.isBound());

        config.runWith(() -> {
            assertTrue(AssistantConfig.isBound());
            assertSame(config, AssistantConfig.current());
        });

        assertFalse(AssistantConfig.isBound());
    }

    @Test
    void testScopedValueConfigCallWith() {
        AssistantConfig config = AssistantConfig.development();
        int result = config.callWith(() -> AssistantConfig.current().vectorDimension());
        assertEquals(64, result);
    }

    // ==================== End-to-End Integration Tests ====================

    @Test
    void testEndToEndSelectQuery() {
        var response = agent.query("select name, email from users where active = true");
        assertNotNull(response);
        assertNotNull(response.query());
        assertNotNull(response.query().sql());
        assertTrue(response.query().sql().contains("SELECT"));
    }

    @Test
    void testEndToEndInsertQuery() {
        var response = agent.query("insert into users values (1, 'test')");
        assertNotNull(response);
        assertTrue(response.query().sql().toUpperCase().contains("INSERT"));
    }

    @Test
    void testEndToEndDeleteQuery() {
        var response = agent.query("delete from orders where id = 5");
        assertNotNull(response);
        assertTrue(response.query().sql().toUpperCase().contains("DELETE"));
    }

    @Test
    void testEndToEndAgentMetrics() {
        agent.query("select * from users");
        agent.query("insert into orders values (1, 2)");

        var metrics = agent.getMetrics();
        assertEquals(2, metrics.get("total_queries"));
        assertEquals(2, metrics.get("successful_queries"));
    }

    @Test
    void testEndToEndConversationState() {
        var state = JakartaAIAdapter.AgentState.initial();
        assertEquals(0, state.interactionCount());

        var response = agent.query("select * from users", state);
        assertEquals(1, response.newState().interactionCount());
    }

    // ==================== ColumnDef Tests ====================

    @Test
    void testColumnDefFactory() {
        ColumnDef col = ColumnDef.of("name", "VARCHAR(255)");
        assertEquals("name", col.name());
        assertFalse(col.primaryKey());
        assertFalse(col.nullable());

        ColumnDef pk = ColumnDef.primaryKey("id", "BIGINT");
        assertTrue(pk.primaryKey());
    }

    @Test
    void testColumnDefValidation() {
        assertThrows(IllegalArgumentException.class, () -> ColumnDef.of("", "INT"));
        assertThrows(IllegalArgumentException.class, () -> ColumnDef.of("name", ""));
    }

    // ==================== IntentNode Tests ====================

    @Test
    void testIntentNodeLeaf() {
        IntentNode node = IntentNode.leaf(
            "test", SemanticHash.ZERO, "Test", "Select", "SELECT * FROM t", 10
        );
        assertEquals("test", node.id());
        assertTrue(node.children().isEmpty());
    }

    @Test
    void testIntentNodeFindChild() {
        IntentNode child = IntentNode.leaf(
            "child1", SemanticHash.ZERO, "Child1", "Select", "SELECT 1", 5
        );
        IntentNode parent = new IntentNode(
            "parent", SemanticHash.ZERO, "Parent", "Unknown", "",
            List.of(child), 0
        );
        assertTrue(parent.findChild("Child1").isPresent());
        assertTrue(parent.findChild("NonExistent").isEmpty());
    }
}
