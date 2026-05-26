#!/bin/bash
# ============================================================
# Export SQL Assistant Enterprise vers GitHub
# ============================================================
# UTILISATION:
#   ./push-to-github.sh <GITHUB_USERNAME> <GITHUB_TOKEN>
#
# Pour creer un token GitHub:
#   1. Allez sur https://github.com/settings/tokens
#   2. Cliquez "Generate new token (classic)"
#   3. Cochez "repo" (full control of private repositories)
#   4. Copiez le token et passez-le en argument
#
# Exemple:
#   ./push-to-github.sh monuser ghp_xxxxxxxxxxxxxxxxxxxx
# ============================================================

set -e

USERNAME="${1:?Usage: $0 <GITHUB_USERNAME> <GITHUB_TOKEN>}"
TOKEN="${2:?Erreur: Token GitHub requis}"

REPO_NAME="sql-assistant-enterprise"
REPO_DESC="Enterprise SQL Assistant - Pure Java, JDK HttpServer, Multi-DB, No LLM, No GPU"

echo "============================================================"
echo "  Export SQL Assistant Enterprise vers GitHub"
echo "============================================================"
echo ""
echo "  Utilisateur : $USERNAME"
echo "  Repository  : $REPO_NAME"
echo ""

# 1. Creer le repository GitHub via API
echo "[1/3] Creation du repository GitHub..."
RESPONSE=$(curl -s -X POST "https://api.github.com/user/repos" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: token $TOKEN" \
  -d "{
    \"name\": \"$REPO_NAME\",
    \"description\": \"$REPO_DESC\",
    \"private\": false,
    \"has_issues\": true,
    \"has_projects\": true,
    \"has_wiki\": true,
    \"auto_init\": false
  }")

# Verifier si le repo a ete cree
HTML_URL=$(echo "$RESPONSE" | grep -o '"html_url":"https://github.com/[^"]*' | head -1 | cut -d'"' -f4)
if [ -z "$HTML_URL" ]; then
  echo "  [!] Le repo existe peut-etre deja ou erreur d'authentification"
  echo "  Reponse: $(echo "$RESPONSE" | head -5)"
  HTML_URL="https://github.com/$USERNAME/$REPO_NAME"
fi
echo "  [OK] Repo: $HTML_URL"

# 2. Configurer le remote
echo ""
echo "[2/3] Configuration du remote Git..."
REMOTE_URL="https://$USERNAME:$TOKEN@github.com/$USERNAME/$REPO_NAME.git"

# Supprimer l'ancien remote si present
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
echo "  [OK] Remote configure"

# 3. Pousser le code
echo ""
echo "[3/3] Push du code vers GitHub..."
git push -u origin main
echo "  [OK] Code pousse avec succes!"

# Nettoyer l'URL du remote (supprimer le token de l'URL)
git remote set-url origin "https://github.com/$USERNAME/$REPO_NAME.git"

echo ""
echo "============================================================"
echo "  SUCCES! Votre projet est maintenant sur GitHub:"
echo "  $HTML_URL"
echo "============================================================"
echo ""
echo "  Clone: git clone https://github.com/$USERNAME/$REPO_NAME.git"
