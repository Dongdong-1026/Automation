#!/usr/bin/env bash
# setup_remote.sh — One-shot post-push setup for HSI Morning Push
#
# Usage:
#   export GOOGLE_CHAT_WEBHOOK='<paste your webhook URL>'
#   export ALLTICK_API_KEY='<your AllTick key>'
#   ./scripts/setup_remote.sh
#
# What it does:
#   1. Checks `gh` is installed and authenticated
#   2. Sets the 3 required GitHub Secrets
#   3. Triggers one manual workflow run for end-to-end verification
#
# Notes:
#   - The webhook URL is read from env, never written to disk or logs.
#   - Requires `gh auth login` to be done first.
set -euo pipefail

REPO="Dongdong-1026/Automation"
SECRETS=(
  "GOOGLE_CHAT_WEBHOOK"
  "ALLTICK_API_KEY"
  "ALLTICK_API_URL"
)
WORKFLOW="daily-morning-push.yml"

# Defaults
: "${ALLTICK_API_URL:=https://api.alltick.co/v1/ohlcv}"

echo "==[1/3] Checking gh CLI =="
if ! command -v gh >/dev/null 2>&1; then
  echo "❌ gh CLI not found. Install with: brew install gh"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "❌ gh not authenticated. Run: gh auth login"
  exit 1
fi
echo "✅ gh authenticated as $(gh api user --jq .login)"

echo ""
echo "==[2/3] Setting Secrets =="
for name in "${SECRETS[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "⚠️  ${name} not set in env; skipping"
    continue
  fi
  gh secret set "$name" --repo "$REPO" --body "${!name}" >/dev/null
  echo "✅ ${name} set"
done

echo ""
echo "==[3/3] Triggering workflow ${WORKFLOW} =="
gh workflow run "$WORKFLOW" --repo "$REPO"
echo "✅ Workflow triggered. View at: https://github.com/${REPO}/actions"

echo ""
echo "Done. Watch the run in the Actions tab."
echo "When complete, check your Automation Space for the Chat card."
