# Enterprise SQL Assistant

> Assistant SQL enterprise 100% Java — sans LLM, sans GPU, sans dependences externes

![Java 21+](https://img.shields.io/badge/Java-21+-orange)
![Pure JDK](https://img.shields.io/badge/Server-JDK_HttpServer-blue)
![H2](https://img.shields.io/badge/DB-H2_Embedded-green)
![No LLM](https://img.shields.io/badge/AI-Local_NLP_No_LLM-red)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CLIENT (HTML/CSS/JS)                │
│  Chat │ Schema Browser │ Execute │ Metrics │ Themes  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/JSON
┌────────────────────▼────────────────────────────────┐
│           JDK HttpServer (com.sun.net.httpserver)     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐│
│  │ /api/     │ │ /api/    │ │ /api/    │ │ /api/   ││
│  │ databases │ │ schema   │ │ query    │ │ execute ││
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └────┬────┘│
│        │             │            │            │      │
│  ┌─────▼─────────────▼────────────▼────────────▼────┐│
│  │            Ensemble SQL Engine (5 strategies)     ││
│  │  Intent Graph │ LSH Multi-Probe │ Syntactic      ││
│  │  Parser │ Thesaurus Hash │ Regex Rules            ││
│  └─────────────────────┬───────────────────────────┘│
│                        │                              │
│  ┌─────────────────────▼───────────────────────────┐│
│  │          Database Manager (Multi-DB JDBC)        ││
│  │  E-Commerce │ Analytics │ RH │ + custom DBs      ││
│  │  Dynamic Schema Discovery (DatabaseMetaData)     ││
│  └─────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

## Fonctionnalites

### Serveur (JDK HttpServer)
- **Multi-database** : configurez plusieurs bases dans `databases.json`
- **Schema dynamique** : decouverte automatique des tables/colonnes via `DatabaseMetaData`
- **Execution SQL** : executez des requetes et retournez les resultats en JSON
- **5 strategies d'ensemble** : vote pondere pour la resolution d'intention
- **Feedback adaptatif** : les poids des strategies s'ajustent selon les retours
- **8 endpoints API** : bases de donnees, schema, query, execute, metrics, strategies, patterns, feedback

### Client (Pure HTML/CSS/JS)
- **Selecteur de base de donnees** dans le header
- **Chat NL -> SQL** : posez une question en langage naturel
- **SQL editable** : modifiez la requete generee avant execution
- **Tableau de resultats** : colonnes, lignes, compteur, temps d'execution
- **Export CSV** : exportez les resultats au format CSV
- **Navigateur de schema** : explorez les tables et colonnes
- **Panel de metriques** : performance des strategies en temps reel
- **Theme sombre/clair** : basculez entre les themes
- **Palette de commandes** : Ctrl+K pour acces rapide

### Moteur NLP (Sans LLM)
- **SimHash 128-bit** : hachage semantique pour similarite
- **HNSW Index** : recherche approximative nearest neighbor
- **Vector API (SIMD)** : operations vectorielles accelerees
- **Panama Memory API** : gestion memoire off-heap
- **DOP (Data-Oriented Programming)** : sealed interfaces, pattern matching
- **Jakarta Agentic AI API** : boucle Perceive->Reason->Act->Reflect

## Demarrage rapide

### Prerequis
- Java 21+ (JDK)
- H2 JDBC Driver (telecharge automatiquement)

### Installation

```bash
# Cloner le depot
git clone https://github.com/VOTRE_USER/sql-assistant-enterprise.git
cd sql-assistant-enterprise

# Telecharger le driver H2
mkdir -p lib
curl -L -o lib/h2-2.2.224.jar https://repo1.maven.org/maven2/com/h2database/h2/2.2.224/h2-2.2.224.jar

# Compiler le serveur
mkdir -p server-build
javac -cp "lib/h2-2.2.224.jar" -d server-build server-src/SqlAssistantServer.java

# Lancer le serveur
chmod +x start.sh
./start.sh
```

### Acces
- **Client** : http://localhost:8080
- **API bases de donnees** : http://localhost:8080/api/databases
- **API schema** : http://localhost:8080/api/schema?db=E-Commerce

## Configuration des bases de donnees

Editez `databases.json` pour ajouter/modifier les bases :

```json
[
  {
    "name": "E-Commerce",
    "description": "Base de donnees e-commerce",
    "driver": "org.h2.Driver",
    "url": "jdbc:h2:mem:ecommerce;DB_CLOSE_DELAY=-1;MODE=PostgreSQL",
    "user": "sa",
    "password": ""
  }
]
```

Supporte tout pilote JDBC (MySQL, PostgreSQL, Oracle, SQL Server...) — il suffit d'ajouter le JAR dans `lib/` et de configurer le driver et l'URL.

## API Endpoints

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/databases` | Liste des bases configurees |
| GET | `/api/schema?db=Nom` | Schema de la base selectionnee |
| POST | `/api/query` | Traduire NL en SQL |
| POST | `/api/execute` | Executer SQL sur une base |
| GET | `/api/metrics` | Metriques de l'agent |
| GET | `/api/strategies` | Poids et precision des strategies |
| GET | `/api/patterns` | Patterns de requetes mines |
| POST | `/api/feedback` | Soumettre un feedback |

## Structure du projet

```
sql-assistant-enterprise/
├── client/
│   └── index.html              # Client HTML/CSS/JS complet
├── server-src/
│   └── SqlAssistantServer.java # Serveur JDK HttpServer
├── src/main/java/io/enterprise/sql/
│   ├── model/                  # Modeles DOP (SqlIntent, SemanticHash...)
│   ├── nlp/                    # NLP sans LLM (Tokenizer, Thesaurus...)
│   ├── hashing/                # SimHash, LSH Multi-Probe
│   ├── vector/                 # HNSW, VectorOps, ProductQuantizer
│   ├── memory/                 # SemanticIndex, MemorySegmentStore
│   ├── intent/                 # IntentResolver, EnsembleResolver
│   ├── sql/                    # SqlGenerator, SchemaResolver
│   ├── jakarta/                # Jakarta Agentic AI adapter
│   └── config/                 # Configuration
├── src/test/java/              # Tests unitaires
├── databases.json              # Configuration multi-DB
├── pom.xml                     # Maven build
└── start.sh                    # Script de demarrage
```

## Technologies

| Composant | Technologie |
|-----------|------------|
| Serveur HTTP | `com.sun.net.httpserver.HttpServer` (JDK) |
| Base de donnees | H2 Embedded (JDBC) |
| Multi-DB | JDBC `DatabaseMetaData` |
| NLP | Tokenizer + Thesaurus + SimHash |
| Vector Search | HNSW + Vector API (SIMD) |
| Memory | Panama Memory API (off-heap) |
| Programming | DOP (sealed interfaces, pattern matching) |
| Agent AI | Jakarta Agentic AI API |

## Licence

MIT
