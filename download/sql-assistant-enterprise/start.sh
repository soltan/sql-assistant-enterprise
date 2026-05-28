#!/bin/bash
# =============================================================
# Enterprise SQL Assistant V4 — Launch Script
# JDK HttpServer Backend (com.sun.net.httpserver) + Multi-DB JDBC + LLM
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAVA_CMD="${JAVA_HOME:-/tmp/jdk-21.0.11+10}/bin/java"
JAVAC_CMD="${JAVA_HOME:-/tmp/jdk-21.0.11+10}/bin/javac"
PORT="${1:-8080}"
CLIENT_DIR="${SCRIPT_DIR}/client"
BUILD_DIR="${SCRIPT_DIR}/server-build"
LIB_DIR="${SCRIPT_DIR}/lib"
CONFIG_FILE="${SCRIPT_DIR}/databases.json"
LLM_CONFIG_FILE="${SCRIPT_DIR}/llm-config.json"
CLASSPATH="${BUILD_DIR}:${LIB_DIR}/h2-2.2.224.jar"

# Always recompile (LLM sources may have changed)
echo "Compiling SqlAssistantServer + LLMAdapter + Qwen35..."
mkdir -p "$BUILD_DIR"
"$JAVAC_CMD" --add-modules=jdk.incubator.vector --enable-preview -cp "${LIB_DIR}/h2-2.2.224.jar" -d "$BUILD_DIR" "$SCRIPT_DIR/server-src/SqlAssistantServer.java" "$SCRIPT_DIR/server-src/LLMAdapter.java" "$SCRIPT_DIR/llm/Qwen35.java"
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "Compilation successful"

echo "Starting Enterprise SQL Assistant V4 on port $PORT..."
echo "Client directory: $CLIENT_DIR"
echo "Config file: $CONFIG_FILE"
echo "LLM config: $LLM_CONFIG_FILE"
echo ""
exec "$JAVA_CMD" --add-modules=jdk.incubator.vector --enable-preview -cp "$CLASSPATH" SqlAssistantServer "$PORT" "$CLIENT_DIR" "$CONFIG_FILE" "$LLM_CONFIG_FILE"
