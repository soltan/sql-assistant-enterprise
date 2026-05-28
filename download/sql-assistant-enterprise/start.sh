#!/bin/bash
# =============================================================
# Enterprise SQL Assistant V5 — Launch Script
# JDK HttpServer Backend (com.sun.net.httpserver) + Multi-DB JDBC + SQL Templates
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAVA_CMD="${JAVA_HOME:-/home/z/.local/jdk}/bin/java"
JAVAC_CMD="${JAVA_HOME:-/home/z/.local/jdk}/bin/javac"
PORT="${1:-8080}"
CLIENT_DIR="${SCRIPT_DIR}/client"
BUILD_DIR="${SCRIPT_DIR}/server-build"
LIB_DIR="${SCRIPT_DIR}/lib"
CONFIG_FILE="${SCRIPT_DIR}/databases.json"
CLASSPATH="${BUILD_DIR}:${LIB_DIR}/h2-2.2.224.jar"

# Compile server
echo "Compiling SqlAssistantServer..."
mkdir -p "$BUILD_DIR"
"$JAVAC_CMD" -cp "${LIB_DIR}/h2-2.2.224.jar" -d "$BUILD_DIR" "$SCRIPT_DIR/server-src/SqlAssistantServer.java"
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "Compilation successful"

echo "Starting Enterprise SQL Assistant V5 on port $PORT..."
echo "Client directory: $CLIENT_DIR"
echo "Config file: $CONFIG_FILE"
echo ""
exec "$JAVA_CMD" -cp "$CLASSPATH" SqlAssistantServer "$PORT" "$CLIENT_DIR" "$CONFIG_FILE"
