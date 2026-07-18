#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_PATH="${1:-$ROOT/.analysis/artifacts/anthropic-adapter-acceptance-$(date -u +%Y%m%dT%H%M%SZ).json}"
CASES_ARG="${2:-${ANTHROPIC_ADAPTER_CASES:-}}"
CONFIG_PATH="${ANTHROPIC_ADAPTER_CONFIG_PATH:-scripts/local-ci/anthropic_adapter_config.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIVELINESS_URL="${LITELLM_DEV_LIVELINESS_URL:-http://127.0.0.1:4001/health/liveliness}"
# Bounded readiness gate (not a fixed sleep): override via env for slow hosts.
READY_MAX_ATTEMPTS="${LITELLM_DEV_READY_MAX_ATTEMPTS:-24}"
READY_INITIAL_DELAY_SECONDS="${LITELLM_DEV_READY_INITIAL_DELAY_SECONDS:-1}"
READY_MAX_DELAY_SECONDS="${LITELLM_DEV_READY_MAX_DELAY_SECONDS:-5}"
READY_CURL_MAX_TIME_SECONDS="${LITELLM_DEV_READY_CURL_MAX_TIME_SECONDS:-2}"

cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export LANGFUSE_QUERY_URL="${LANGFUSE_QUERY_URL:-http://127.0.0.1:3000}"

# Probe /health/liveliness with bounded retry + exponential backoff.
# Does not start or restart containers; optional compose start is outside this helper.
wait_for_litellm_dev_ready() {
  local url="$LIVELINESS_URL"
  local max_attempts="$READY_MAX_ATTEMPTS"
  local delay_seconds="$READY_INITIAL_DELAY_SECONDS"
  local max_delay_seconds="$READY_MAX_DELAY_SECONDS"
  local curl_max_time="$READY_CURL_MAX_TIME_SECONDS"
  local attempt=1
  local current_delay

  # Normalize to non-negative integers (portable for bash arithmetic).
  case "$max_attempts" in
    ''|*[!0-9]*) max_attempts=24 ;;
  esac
  case "$delay_seconds" in
    ''|*[!0-9]*) delay_seconds=1 ;;
  esac
  case "$max_delay_seconds" in
    ''|*[!0-9]*) max_delay_seconds=5 ;;
  esac
  case "$curl_max_time" in
    ''|*[!0-9]*) curl_max_time=2 ;;
  esac
  if [[ "$max_attempts" -lt 1 ]]; then
    max_attempts=1
  fi
  if [[ "$delay_seconds" -lt 0 ]]; then
    delay_seconds=0
  fi
  if [[ "$max_delay_seconds" -lt "$delay_seconds" ]]; then
    max_delay_seconds="$delay_seconds"
  fi
  current_delay="$delay_seconds"

  echo "Waiting for litellm-dev readiness at ${url} (max ${max_attempts} attempts)..."
  while [[ "$attempt" -le "$max_attempts" ]]; do
    if curl -sf --max-time "$curl_max_time" "$url" >/dev/null 2>&1; then
      echo "litellm-dev is healthy (attempt ${attempt}/${max_attempts})"
      return 0
    fi
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      break
    fi
    echo "litellm-dev not ready yet (attempt ${attempt}/${max_attempts}); retrying in ${current_delay}s..."
    sleep "$current_delay"
    # Exponential backoff, capped.
    current_delay=$((current_delay * 2))
    if [[ "$current_delay" -gt "$max_delay_seconds" ]]; then
      current_delay="$max_delay_seconds"
    fi
    attempt=$((attempt + 1))
  done

  echo "litellm-dev is not healthy on :4001 after ${max_attempts} attempt(s) (${url})"
  return 1
}

docker_status="$(docker ps --filter name=^litellm-dev$ --format '{{.Status}}' || true)"
if [[ -z "$docker_status" ]]; then
  echo "litellm-dev is not running; starting it now..."
  docker compose -f docker-compose.dev.yml up -d litellm-dev
fi

if ! wait_for_litellm_dev_ready; then
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
