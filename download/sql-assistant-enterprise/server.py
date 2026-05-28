#!/usr/bin/env python3
"""
Enterprise SQL Assistant — Backend Server
==========================================
Serveur HTTP qui expose l'API REST du pipeline SQL Assistant.

Architecture (mirrors the Java backend):
  1. PERCEIVE  — Normalisation de l'entree
  2. REASON    — Resolution d'intent via 5 strategies ensemble
  3. ACT       — Generation SQL
  4. REFLECT   — Validation

Endpoints:
  GET  /              → Client HTML
  POST /api/query     → Traduire du NL en SQL
  GET  /api/schema    → Schema des tables
  GET  /api/metrics   → Metriques de l'agent
  GET  /api/strategies→ Poids et accuracy des strategies
  GET  /api/patterns  → Patterns mines
  POST /api/feedback  → Feedback sur une requete
"""

import json
import re
import time
import math
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

SCHEMA = {
    "users": {
        "columns": [
            {"name": "id", "type": "BIGINT", "pk": True, "nullable": False},
            {"name": "username", "type": "VARCHAR(255)", "pk": False, "nullable": False},
            {"name": "email", "type": "VARCHAR(512)", "pk": False, "nullable": False},
            {"name": "created_at", "type": "TIMESTAMP", "pk": False, "nullable": True},
            {"name": "active", "type": "BOOLEAN", "pk": False, "nullable": True}
        ],
        "indexes": ["idx_users_username", "idx_users_email"]
    },
    "orders": {
        "columns": [
            {"name": "id", "type": "BIGINT", "pk": True, "nullable": False},
            {"name": "user_id", "type": "BIGINT", "pk": False, "nullable": False},
            {"name": "product_id", "type": "BIGINT", "pk": False, "nullable": False},
            {"name": "quantity", "type": "INTEGER", "pk": False, "nullable": False},
            {"name": "total_price", "type": "DECIMAL(10,2)", "pk": False, "nullable": False},
            {"name": "order_date", "type": "TIMESTAMP", "pk": False, "nullable": False},
            {"name": "status", "type": "VARCHAR(50)", "pk": False, "nullable": False}
        ],
        "indexes": ["idx_orders_user_id", "idx_orders_status"]
    },
    "products": {
        "columns": [
            {"name": "id", "type": "BIGINT", "pk": True, "nullable": False},
            {"name": "name", "type": "VARCHAR(512)", "pk": False, "nullable": False},
            {"name": "description", "type": "TEXT", "pk": False, "nullable": True},
            {"name": "price", "type": "DECIMAL(10,2)", "pk": False, "nullable": False},
            {"name": "category", "type": "VARCHAR(100)", "pk": False, "nullable": True},
            {"name": "stock", "type": "INTEGER", "pk": False, "nullable": False}
        ],
        "indexes": ["idx_products_category"]
    }
}

INTENT_COLORS = {
    "Select": "#3b82f6", "Insert": "#10b981", "Update": "#f59e0b", "Delete": "#ef4444",
    "CreateTable": "#8b5cf6", "AlterTable": "#06b6d4", "DropTable": "#ec4899",
    "Join": "#6366f1", "Aggregate": "#14b8a6", "Subquery": "#a855f7",
    "Explain": "#0ea5e9", "SchemaInfo": "#84cc16", "Unknown": "#6b7280"
}

STRATEGY_COLORS = {
    "intent_graph": "#3b82f6", "lsh_multi_probe": "#8b5cf6",
    "syntactic_parser": "#10b981", "thesaurus_hash": "#f59e0b",
    "regex_rules": "#ef4444"
}

# ═══════════════════════════════════════════════════════════════
# SIMULATION ENGINE (Mirrors the Java backend pipeline)
# ═══════════════════════════════════════════════════════════════

class SqlAssistantEngine:
    """Server-side engine that mirrors the Java EnhancedSqlAssistant pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {
            "total_queries": 0, "successful_queries": 0,
            "total_latency_nanos": 0, "correct_feedback": 0, "incorrect_feedback": 0
        }
        self.strategy_weights = {
            "intent_graph": 1.0, "lsh_multi_probe": 0.9,
            "syntactic_parser": 1.2, "thesaurus_hash": 0.8, "regex_rules": 0.6
        }
        self.strategy_accuracy = {
            "intent_graph": 0.88, "lsh_multi_probe": 0.82,
            "syntactic_parser": 0.92, "thesaurus_hash": 0.78, "regex_rules": 0.70
        }
        self.patterns = []
        self.history = []

    def resolve_intent(self, query):
        """5-strategy ensemble resolution (mirrors EnsembleResolver.java)."""
        q = query.lower()
        votes = []
        weights = self.strategy_weights

        # Strategy 1: Intent Graph (keyword hash → graph traversal)
        ig_intent, ig_conf = "Unknown", 0.1
        if re.search(r'\b(select|show|get|find|list|display|montre|affiche|cherche|liste)\b', q) and re.search(r'\b(from|de|dans)\b', q):
            ig_intent, ig_conf = "Select", 0.85
        elif re.search(r'\b(how many|combien|count|nombre)\b', q):
            ig_intent, ig_conf = "Aggregate", 0.80
        elif re.search(r'\b(total|somme|sum)\b', q):
            ig_intent, ig_conf = "Aggregate", 0.75
        elif re.search(r'\b(insert|add|ajoute|ajouter|creer un)\b', q):
            ig_intent, ig_conf = "Insert", 0.85
        elif re.search(r'\b(update|modify|change|modifie|modifier|mettre a jour)\b', q):
            ig_intent, ig_conf = "Update", 0.80
        elif re.search(r'\b(delete|remove|supprime|supprimer|efface)\b', q):
            ig_intent, ig_conf = "Delete", 0.85
        elif re.search(r'\b(create table|cre.*table|nouvelle table)\b', q):
            ig_intent, ig_conf = "CreateTable", 0.90
        elif re.search(r'\b(alter|modifier.*table)\b', q):
            ig_intent, ig_conf = "AlterTable", 0.90
        elif re.search(r'\b(drop|supprimer.*table)\b', q):
            ig_intent, ig_conf = "DropTable", 0.90
        elif re.search(r'\b(join|combine|fusionne|joindre|combiner)\b', q):
            ig_intent, ig_conf = "Join", 0.70
        elif re.search(r'\b(explain|analyse|explique)\b', q):
            ig_intent, ig_conf = "Explain", 0.90
        elif re.search(r'\b(describe|schema|structure|decris|colonnes)\b', q):
            ig_intent, ig_conf = "SchemaInfo", 0.80
        votes.append({"strategy": "intent_graph", "intentType": ig_intent,
                       "rawConfidence": ig_conf, "weightedScore": ig_conf * weights["intent_graph"]})

        # Strategy 2: LSH Multi-Probe (8 tables × 5 probes)
        lsh_intent, lsh_conf = "Unknown", 0.1
        if re.search(r'\b(select|show|get|affiche|montre|liste|cherche)\b', q):
            lsh_intent, lsh_conf = "Select", 0.80
        elif re.search(r'\b(count|combien|nombre|total|somme)\b', q):
            lsh_intent, lsh_conf = "Aggregate", 0.75
        elif re.search(r'\b(insert|ajout|add)\b', q):
            lsh_intent, lsh_conf = "Insert", 0.82
        elif re.search(r'\b(update|modif|change)\b', q):
            lsh_intent, lsh_conf = "Update", 0.78
        elif re.search(r'\b(delete|supprim|remove)\b', q):
            lsh_intent, lsh_conf = "Delete", 0.82
        elif re.search(r'\b(join|combin|fusion)\b', q):
            lsh_intent, lsh_conf = "Join", 0.68
        votes.append({"strategy": "lsh_multi_probe", "intentType": lsh_intent,
                       "rawConfidence": lsh_conf, "weightedScore": lsh_conf * weights["lsh_multi_probe"]})

        # Strategy 3: Syntactic Parser (SVO extraction)
        syn_intent, syn_conf = "Unknown", 0.1
        verb_map = {
            "select": "Select", "get": "Select", "show": "Select", "find": "Select",
            "list": "Select", "affiche": "Select", "montre": "Select", "cherche": "Select",
            "insert": "Insert", "add": "Insert", "ajoute": "Insert", "ajouter": "Insert",
            "update": "Update", "modify": "Update", "change": "Update",
            "modifie": "Update", "modifier": "Update",
            "delete": "Delete", "remove": "Delete", "supprime": "Delete",
            "supprimer": "Delete", "efface": "Delete",
            "create": "CreateTable", "creer": "CreateTable",
            "join": "Join", "combine": "Join", "fusionne": "Join", "joindre": "Join",
            "explain": "Explain", "explique": "Explain", "analyse": "Explain",
            "describe": "SchemaInfo", "decris": "SchemaInfo"
        }
        for word in q.split():
            if word in verb_map:
                syn_intent, syn_conf = verb_map[word], 0.88
                break
        if syn_intent == "Unknown" and re.search(r'\b(combien|nombre|count|total|moyenne|average|somme|sum|max|min)\b', q):
            syn_intent, syn_conf = "Aggregate", 0.85
        votes.append({"strategy": "syntactic_parser", "intentType": syn_intent,
                       "rawConfidence": syn_conf, "weightedScore": syn_conf * weights["syntactic_parser"]})

        # Strategy 4: Thesaurus-expanded SimHash
        thes_intent, thes_conf = "Unknown", 0.1
        thes_map = {
            "utilisateurs": "Select", "clients": "Select", "commandes": "Select",
            "produits": "Select", "donnees": "Select", "enregistrements": "Select",
            "nombre": "Aggregate", "total": "Aggregate", "moyenne": "Aggregate",
            "somme": "Aggregate", "ajouter": "Insert", "inserer": "Insert",
            "modifier": "Update", "supprimer": "Delete", "joindre": "Join",
            "combiner": "Join", "fusionner": "Join", "expliquer": "Explain",
            "decrire": "SchemaInfo"
        }
        for k, v in thes_map.items():
            if k in q:
                thes_intent, thes_conf = v, 0.76
                break
        votes.append({"strategy": "thesaurus_hash", "intentType": thes_intent,
                       "rawConfidence": thes_conf, "weightedScore": thes_conf * weights["thesaurus_hash"]})

        # Strategy 5: Regex Rules
        reg_intent, reg_conf = "Unknown", 0.1
        rules = [
            (r'\b(select|show|get|find|list|display)\b.*\b(from|in)\b', "Select", 0.85),
            (r'\b(insert|add|put)\b.*\b(into|to|in)\b', "Insert", 0.85),
            (r'\b(update|modify|change)\b.*\b(set)\b', "Update", 0.85),
            (r'\b(delete|remove)\b.*\b(from)\b', "Delete", 0.85),
            (r'\b(create|make)\s+(table)\b', "CreateTable", 0.9),
            (r'\b(alter|modify)\s+(table)\b', "AlterTable", 0.9),
            (r'\b(drop|remove)\s+(table)\b', "DropTable", 0.9),
            (r'\b(join|combine|merge)\b', "Join", 0.7),
            (r'\b(count|sum|average|avg|min|max|how many|combien)\b', "Aggregate", 0.8),
            (r'\b(explain|analyse)\b', "Explain", 0.9),
            (r'\b(describe|schema|colonnes)\b', "SchemaInfo", 0.8),
            (r'\b(montre|affiche|liste|cherche)\b', "Select", 0.8),
            (r'\b(ajoute|insere|creer)\b', "Insert", 0.8),
            (r'\b(modifie|change|met a jour)\b', "Update", 0.8),
            (r'\b(supprime|efface|retire)\b', "Delete", 0.8),
            (r'\b(combien|nombre|total|somme|moyenne)\b', "Aggregate", 0.8),
            (r'\b(joindre|combiner|fusionner)\b', "Join", 0.75),
            (r'\b(explique|analyse)\b', "Explain", 0.85),
            (r'\b(decris|structure|colonnes de)\b', "SchemaInfo", 0.8),
        ]
        for pattern, intent, conf in rules:
            if re.search(pattern, q):
                reg_intent, reg_conf = intent, conf
                break
        votes.append({"strategy": "regex_rules", "intentType": reg_intent,
                       "rawConfidence": reg_conf, "weightedScore": reg_conf * weights["regex_rules"]})

        # Aggregate votes
        scores = {}
        best_raw = {}
        for v in votes:
            scores[v["intentType"]] = scores.get(v["intentType"], 0) + v["weightedScore"]
            best_raw[v["intentType"]] = max(best_raw.get(v["intentType"], 0), v["rawConfidence"])

        winner = max(scores.items(), key=lambda x: x[1])
        total = sum(scores.values())
        ensemble_conf = winner[1] / total if total > 0 else 0
        calibrated = ensemble_conf * 0.6 + best_raw.get(winner[0], 0.5) * 0.4

        return {
            "intent": winner[0],
            "confidence": round(calibrated, 4),
            "ensembleConfidence": round(ensemble_conf, 4),
            "votes": votes,
            "scores": {k: round(v, 4) for k, v in scores.items()}
        }

    def generate_sql(self, query, intent):
        """SQL generator (mirrors SqlGenerator.java pattern matching)."""
        q = query.lower()

        def resolve_table(s):
            tmap = {
                "users": "users", "user": "users", "utilisateur": "users", "utilisateurs": "users",
                "orders": "orders", "order": "orders", "commande": "orders", "commandes": "orders",
                "products": "products", "product": "products", "produit": "products", "produits": "products"
            }
            for k, v in tmap.items():
                if k in s:
                    return v
            return list(SCHEMA.keys())[0]

        def extract_conditions():
            conds = []
            if re.search(r'\b(active|actif)\b', q):
                conds.append("active = true")
            if re.search(r'\b(inactive|inactif)\b', q):
                conds.append("active = false")
            m = re.search(r'(\d+)', q)
            if re.search(r'greater than|plus de|superieur|>\s*\d+|more than', q) and m:
                conds.append(f"price > {m.group(1)}")
            if re.search(r'less than|moins de|inferieur|<\s*\d+', q) and m:
                conds.append(f"price < {m.group(1)}")
            if re.search(r'\b(recent|recente|dernier|last)\b', q):
                conds.append("created_at >= CURRENT_DATE - INTERVAL '30 days'")
            sm = re.search(r'(?:status|etat)\s*(?:=|est)\s*[\'"]?(\w+)', q)
            if sm:
                conds.append(f"status = '{sm.group(1)}'")
            return conds

        if intent == "Select":
            table = resolve_table(q)
            if re.search(r'\b(all|tous|tout|\*)\b', q):
                cols = "*"
            else:
                cols = ", ".join(c["name"] for c in SCHEMA[table]["columns"])
            conds = extract_conditions()
            where = "\nWHERE " + "\n  AND ".join(conds) if conds else ""
            om = re.search(r'\b(order by|trie|sorted|tri)\s+(?:by\s+)?(\w+)', q)
            order = f"\nORDER BY {om.group(2)}" if om else ""
            lm = re.search(r'\b(top|limit|premier)\s+(\d+)', q)
            limit = f"\nLIMIT {lm.group(2)}" if lm else ""
            return f"SELECT {cols}\nFROM {table}{where}{order}{limit};"

        elif intent == "Insert":
            table = resolve_table(q)
            cols = [c["name"] for c in SCHEMA[table]["columns"] if not c["pk"]]
            vals = ", ".join(f":{c}" for c in cols)
            return f"INSERT INTO {table} ({', '.join(cols)})\nVALUES ({vals});"

        elif intent == "Update":
            table = resolve_table(q)
            conds = extract_conditions()
            where = "\nWHERE " + "\n  AND ".join(conds) if conds else "\nWHERE id = :id"
            return f"UPDATE {table}\nSET :column = :value{where};"

        elif intent == "Delete":
            table = resolve_table(q)
            conds = extract_conditions()
            where = "\nWHERE " + "\n  AND ".join(conds) if conds else "\nWHERE id = :id"
            return f"DELETE FROM {table}{where};"

        elif intent == "CreateTable":
            nm = re.search(r'(?:table)\s+(\w+)', q)
            name = nm.group(1) if nm else "new_table"
            return f"CREATE TABLE {name} (\n  id BIGINT PRIMARY KEY,\n  name VARCHAR(255) NOT NULL,\n  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);"

        elif intent == "Join":
            tables = list(SCHEMA.keys())
            return (f"SELECT u.username, o.total_price, o.order_date\n"
                    f"FROM {tables[0]} u\n"
                    f"INNER JOIN {tables[1]} o ON u.id = o.user_id\n"
                    f"INNER JOIN {tables[2]} p ON o.product_id = p.id;")

        elif intent == "Aggregate":
            table = resolve_table(q)
            fn = "COUNT"
            if re.search(r'\b(sum|total|somme)\b', q):
                fn = "SUM"
            elif re.search(r'\b(average|avg|moyenne|mean)\b', q):
                fn = "AVG"
            elif re.search(r'\b(max|maximum|highest|plus grand)\b', q):
                fn = "MAX"
            elif re.search(r'\b(min|minimum|lowest|plus petit)\b', q):
                fn = "MIN"
            gm = re.search(r'\b(group by|par|by)\s+(\w+)', q)
            group = f"\nGROUP BY {gm.group(2)}" if gm else ""
            conds = extract_conditions()
            where = "\nWHERE " + "\n  AND ".join(conds) if conds else ""
            col = gm.group(2) if gm else "1"
            return f"SELECT {fn}(*), {col}\nFROM {table}{where}{group};"

        elif intent == "Explain":
            return f"EXPLAIN ANALYZE\nSELECT * FROM {resolve_table(q)} WHERE id = :id;"

        elif intent == "SchemaInfo":
            table = resolve_table(q)
            cols = SCHEMA[table]["columns"]
            lines = []
            for c in cols:
                s = f"  {c['name']} {c['type']}"
                if c["pk"]:
                    s += " PRIMARY KEY"
                if not c["nullable"]:
                    s += " NOT NULL"
                lines.append(s)
            return f"-- Schema: {table}\n-- Columns:\n" + ",\n".join(lines)

        else:
            return f"-- Unable to resolve intent for: \"{query}\"\n-- Please rephrase your query."

    def process_query(self, query):
        """Full pipeline: Perceive → Reason → Act → Reflect."""
        start = time.perf_counter_ns()

        # Step 1: PERCEIVE
        normalized = query.strip()
        if not normalized:
            return {"error": "Query must not be blank"}

        # Step 2: REASON (ensemble)
        result = self.resolve_intent(normalized)

        # Step 3: ACT (generate SQL)
        sql = self.generate_sql(normalized, result["intent"])

        # Step 4: REFLECT (validate)
        valid = not sql.startswith("--") and len(sql) > 5
        validation_msg = "OK" if valid else "Generated SQL has issues"

        latency = time.perf_counter_ns() - start
        latency_us = latency // 1000

        # Record metrics
        with self.lock:
            query_id = len(self.history)
            self.metrics["total_queries"] += 1
            self.metrics["successful_queries"] += 1
            self.metrics["total_latency_nanos"] += latency
            self.history.append({
                "id": query_id,
                "query": query,
                "intent": result["intent"],
                "sql": sql,
                "confidence": result["confidence"],
                "latency_us": latency_us,
                "valid": valid,
                "votes": result["votes"],
                "time": time.time()
            })

            # Pattern mining
            existing = next((p for p in self.patterns if p["intent"] == result["intent"]), None)
            if existing:
                existing["freq"] += 1
                existing["avgConf"] = (existing["avgConf"] + result["confidence"]) / 2
            else:
                self.patterns.append({"intent": result["intent"], "freq": 1, "avgConf": result["confidence"]})
            self.patterns.sort(key=lambda p: p["freq"], reverse=True)

        return {
            "queryId": query_id,
            "query": query,
            "intent": result["intent"],
            "confidence": result["confidence"],
            "ensembleConfidence": result["ensembleConfidence"],
            "sql": sql,
            "valid": valid,
            "validationMessage": validation_msg,
            "latencyMicros": latency_us,
            "votes": result["votes"],
            "scores": result["scores"]
        }

    def record_feedback(self, query_id, is_correct):
        """Record feedback and adapt strategy weights (mirrors AdaptiveConfidence.java)."""
        with self.lock:
            if is_correct:
                self.metrics["correct_feedback"] += 1
            else:
                self.metrics["incorrect_feedback"] += 1

            if query_id < len(self.history):
                h = self.history[query_id]
                for v in h.get("votes", []):
                    strategy = v["strategy"]
                    current_w = self.strategy_weights.get(strategy, 1.0)
                    if is_correct:
                        self.strategy_weights[strategy] = min(2.0, current_w * 1.05)
                    else:
                        self.strategy_weights[strategy] = max(0.3, current_w * 0.95)

                    acc = self.strategy_accuracy.get(strategy, 0.5)
                    self.strategy_accuracy[strategy] = acc * 0.9 + (1.0 if is_correct else 0.0) * 0.1

    def get_metrics(self):
        with self.lock:
            m = self.metrics.copy()
            avg = m["total_latency_nanos"] // max(m["total_queries"], 1) // 1000
            total_fb = m["correct_feedback"] + m["incorrect_feedback"]
            accuracy = round(m["correct_feedback"] / total_fb * 100) if total_fb > 0 else 0
            return {
                "totalQueries": m["total_queries"],
                "successfulQueries": m["successful_queries"],
                "avgLatencyMicros": avg,
                "accuracy": accuracy,
                "correctFeedback": m["correct_feedback"],
                "incorrectFeedback": m["incorrect_feedback"]
            }

    def get_strategies(self):
        with self.lock:
            return {
                "weights": {k: round(v, 3) for k, v in self.strategy_weights.items()},
                "accuracy": {k: round(v, 4) for k, v in self.strategy_accuracy.items()}
            }

    def get_patterns(self):
        with self.lock:
            return self.patterns[:20]


# ═══════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════

engine = SqlAssistantEngine()
CLIENT_DIR = Path(__file__).parent / "client"

class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self.serve_file(CLIENT_DIR / "index.html", "text/html")
        elif path == "/api/schema":
            self.send_json(SCHEMA)
        elif path == "/api/metrics":
            self.send_json(engine.get_metrics())
        elif path == "/api/strategies":
            self.send_json(engine.get_strategies())
        elif path == "/api/patterns":
            self.send_json(engine.get_patterns())
        elif path == "/api/history":
            with engine.lock:
                self.send_json(engine.history[-50:])
        else:
            # Try serving static files from client dir
            file_path = CLIENT_DIR / path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                ext = file_path.suffix.lower()
                ct = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
                      ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml"
                      }.get(ext, "application/octet-stream")
                self.serve_file(file_path, ct)
            else:
                self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len else ""

        if path == "/api/query":
            try:
                data = json.loads(body) if body else {}
                query = data.get("query", "").strip()
                if not query:
                    self.send_json({"error": "Query must not be blank"}, 400)
                    return
                result = engine.process_query(query)
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif path == "/api/feedback":
            try:
                data = json.loads(body) if body else {}
                query_id = data.get("queryId", -1)
                is_correct = data.get("correct", False)
                engine.record_feedback(query_id, is_correct)
                self.send_json({"status": "ok"})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404)

    def log_message(self, format, *args):
        """Quieter logging."""
        pass


def main():
    host = "0.0.0.0"
    port = 8080
    print("=" * 60)
    print("  Enterprise SQL Assistant V2 — Backend Server")
    print("  Pure Python (mirrors Java 26 + Jakarta AI pipeline)")
    print("=" * 60)
    print(f"  Endpoints:")
    print(f"    GET  /              → Client HTML")
    print(f"    POST /api/query     → Translate NL → SQL")
    print(f"    GET  /api/schema    → Schema metadata")
    print(f"    GET  /api/metrics   → Agent metrics")
    print(f"    GET  /api/strategies→ Strategy weights/accuracy")
    print(f"    GET  /api/patterns  → Mined patterns")
    print(f"    POST /api/feedback  → Submit feedback")
    print(f"")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    server = HTTPServer((host, port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
