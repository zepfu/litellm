#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_PATH="${1:-$ROOT/.analysis/artifacts/anthropic-adapter-acceptance-$(date -u +%Y%m%dT%H%M%SZ).json}"
CASES_ARG="${2:-${ANTHROPIC_ADAPTER_CASES:-}}"
CONFIG_PATH="${ANTHROPIC_ADAPTER_CONFIG_PATH:-scripts/local-ci/anthropic_adapter_config.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export LANGFUSE_QUERY_URL="${LANGFUSE_QUERY_URL:-http://127.0.0.1:3000}"

docker_status="$(docker ps --filter name=^litellm-dev$ --format '{{.Status}}' || true)"
if [[ -z "$docker_status" ]]; then
  echo "litellm-dev is not running; starting it now..."
  docker compose -f docker-compose.dev.yml up -d litellm-dev
  sleep 5
fi

if ! curl -sf http://127.0.0.1:4001/health/liveliness >/dev/null; then
  echo "litellm-dev is not healthy on :4001"
  exit 1
fi

ARGS=(
  scripts/local-ci/run_anthropic_adapter_acceptance.py
  --config "$CONFIG_PATH"
  --write-artifact "$ARTIFACT_PATH"
)
if [[ -n "$CASES_ARG" ]]; then
  ARGS+=(--cases "$CASES_ARG")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
