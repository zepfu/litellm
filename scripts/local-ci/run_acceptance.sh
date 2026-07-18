#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_PATH="${1:-$ROOT/.analysis/artifacts/local-acceptance-$(date -u +%Y%m%dT%H%M%SZ).json}"
CLI_CLAUDE_FANOUT_MODE="${2:-}"
REBUILD_LITELLM_DEV="${REBUILD_LITELLM_DEV:-1}"
BUILD_STATE_PATH="${BUILD_STATE_PATH:-$ROOT/.analysis/artifacts/litellm-dev-build-state.json}"
CONFIG_PATH="${ACCEPTANCE_CONFIG_PATH:-scripts/local-ci/config.json}"

cd "$ROOT"

# RR-081 Medium #3: do NOT `set -a; source .env` (that re-exports every secret
# into the process environment inherited by provider CLIs). Load only the keys
# the harness parent process needs; run_acceptance.py further scrubs child CLI
# envs via `_scrubbed_child_env`.
load_harness_dotenv() {
  local env_file="${1:-.env}"
  [[ -f "$env_file" ]] || return 0

  local line key value
  # Prefixes required by the Python harness for Langfuse polling / URLs.
  local -a allow_prefixes=(
    "LANGFUSE_"
  )
  # Exact keys that may legitimately configure the harness wrapper / overrides.
  local -a allow_keys=(
    "LITELLM_BASE_URL"
    "LITELLM_PORT"
    "ACCEPTANCE_CONFIG_PATH"
    "ACCEPTANCE_CLI_OUTPUT_MAX_CHARS"
    "CLAUDE_FANOUT_MODE"
    "REBUILD_LITELLM_DEV"
    "BUILD_STATE_PATH"
    "PYTHON_BIN"
    "AAWM_HARNESS_RUN_ID"
    "AAWM_HARNESS_USER_ID"
    "AAWM_CLAUDE_HARNESS_USER_ID"
    "AAWM_OBSERVE_SERVICE_NAME"
    "PYTEST_CLASSIFIER_HARNESS_USER_ID"
    "PYTEST_CLASSIFIER_ENABLE_OBSERVABILITY"
  )

  is_harness_env_key() {
    local candidate="$1"
    local allowed prefix
    for allowed in "${allow_keys[@]}"; do
      if [[ "$candidate" == "$allowed" ]]; then
        return 0
      fi
    done
    for prefix in "${allow_prefixes[@]}"; do
      if [[ "$candidate" == "$prefix"* ]]; then
        return 0
      fi
    done
    return 1
  }

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Strip CR (Windows line endings) and leading/trailing whitespace.
    line="${line//$''/}"
    # Trim leading whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == export[[:space:]]* ]]; then
      line="${line#export}"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    # Only KEY=VALUE assignments; skip shell syntax we do not evaluate.
    [[ "$line" == *"="* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    # Trim whitespace around key.
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    is_harness_env_key "$key" || continue

    # Strip matching single/double quotes around the value.
    if [[ ${#value} -ge 2 ]]; then
      if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    # Export only allowlisted keys (no `set -a` / full-file source).
    export "$key=$value"
  done < "$env_file"
}

load_harness_dotenv "$ROOT/.env"

export LANGFUSE_QUERY_URL="${LANGFUSE_QUERY_URL:-http://127.0.0.1:3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLAUDE_FANOUT_MODE="${CLI_CLAUDE_FANOUT_MODE:-${CLAUDE_FANOUT_MODE:-minimal}}"

resolve_litellm_base_url() {
  "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import json
import os
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
config = json.loads(config_path.read_text())
print(os.environ.get("LITELLM_BASE_URL") or config.get("litellm_base_url", "http://127.0.0.1:4001"))
PY
}

compute_build_fingerprint() {
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
# Heavy / volatile trees that must not affect rebuild fingerprint I/O or churn.
exclude_prefixes = [
    ".git/",
    ".analysis/",
    "captures/",
    "scripts/local-ci/",
    ".gemini/",
    ".codex/",
    ".venv/",
    "venv/",
    "node_modules/",
    "ui/litellm-dashboard/node_modules/",
    "ui/litellm-dashboard/.next/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    "dist/",
    "build/",
    ".wheel-build/",
    "htmlcov/",
    ".coverage/",
    "docs/my-website/node_modules/",
    "docs/my-website/build/",
    "docs/my-website/.docusaurus/",
]
exclude_exact = {
    "langfuse-traces.png",
    "litellm/integrations/aawm_agent_identity.py",
    "litellm/integrations/aawm_payload_capture.py",
    "litellm-dev-config.yaml",
    ".env",
    ".env.local",
}
exclude_suffixes = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".egg-info",
)
exclude_name_parts = (
    "/__pycache__/",
    "/node_modules/",
    "/.venv/",
    "/venv/",
    "/.pytest_cache/",
    "/.mypy_cache/",
    "/.ruff_cache/",
    "/.git/",
    "/dist/",
    "/build/",
)

def include(path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in exclude_exact:
        return False
    if any(rel.startswith(prefix) for prefix in exclude_prefixes):
        return False
    # Nested vendor/cache dirs anywhere in the tree (e.g. package/node_modules/).
    if any(part in f"/{rel}/" or part in f"/{rel}" for part in exclude_name_parts):
        return False
    if any(rel.endswith(suffix) for suffix in exclude_suffixes):
        return False
    # Drop compiled/cache artifacts by basename.
    name = path.name
    if name == ".env" or name.startswith(".env."):
        return False
    if name.endswith(".pyc") or name == "__pycache__":
        return False
    return True

sha = hashlib.sha256()
files = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if not include(path):
        continue
    rel = path.relative_to(root).as_posix()
    files.append(rel)
    sha.update(rel.encode("utf-8"))
    sha.update(b"\0")
    sha.update(path.read_bytes())
    sha.update(b"\0")

print(json.dumps({"fingerprint": sha.hexdigest(), "file_count": len(files)}))
PY
}

should_rebuild_litellm_dev() {
  # Decision only — never persist BUILD_STATE_PATH here. Writing the fingerprint
  # before docker compose succeeds would skip rebuilds after a failed build (RR-081).
  local state_json current_fingerprint previous_fingerprint image_present container_present
  state_json="$(compute_build_fingerprint)"
  current_fingerprint="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read())["fingerprint"])' <<<"$state_json")"
  # Export for caller to persist only after a successful rebuild.
  LITELLM_DEV_BUILD_STATE_JSON="$state_json"
  export LITELLM_DEV_BUILD_STATE_JSON

  previous_fingerprint=""
  if [[ -f "$BUILD_STATE_PATH" ]]; then
    previous_fingerprint="$("$PYTHON_BIN" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); print(json.loads(p.read_text()).get("fingerprint",""))' "$BUILD_STATE_PATH" 2>/dev/null || true)"
  fi

  image_present="$(docker images -q litellm-litellm-dev 2>/dev/null || true)"
  container_present="$(docker ps -a --filter name=^litellm-dev$ --format '{{.ID}}' 2>/dev/null || true)"

  if [[ -z "$image_present" || -z "$container_present" ]]; then
    return 0
  fi

  if [[ "$current_fingerprint" != "$previous_fingerprint" ]]; then
    return 0
  fi

  return 1
}

persist_litellm_dev_build_state() {
  mkdir -p "$(dirname "$BUILD_STATE_PATH")"
  if [[ -n "${LITELLM_DEV_BUILD_STATE_JSON:-}" ]]; then
    printf '%s\n' "$LITELLM_DEV_BUILD_STATE_JSON" > "$BUILD_STATE_PATH"
  else
    compute_build_fingerprint > "$BUILD_STATE_PATH"
  fi
}

TARGET_LITELLM_BASE_URL="$(resolve_litellm_base_url)"

if [[ "$TARGET_LITELLM_BASE_URL" == "http://127.0.0.1:4001" ]]; then
  if [[ "$REBUILD_LITELLM_DEV" == "1" ]] && should_rebuild_litellm_dev; then
    echo "litellm-dev image inputs changed; rebuilding and recreating the container..."
    docker compose -f docker-compose.dev.yml build litellm-dev
    docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev
    # Only record fingerprint after successful build+recreate (set -e aborts otherwise).
    persist_litellm_dev_build_state
    echo "Waiting briefly for litellm-dev to finish booting..."
    sleep 10
  elif [[ "$REBUILD_LITELLM_DEV" == "1" ]]; then
    echo "litellm-dev image inputs unchanged; skipping rebuild."
    docker_status="$(docker ps --filter name=litellm-dev --format '{{.Status}}' || true)"
    if [[ -z "$docker_status" ]]; then
      echo "litellm-dev is not running; starting existing image..."
      docker compose -f docker-compose.dev.yml up -d litellm-dev
      echo "Waiting briefly for litellm-dev to finish booting..."
      sleep 10
    fi
  else
    docker_status="$(docker ps --filter name=litellm-dev --format '{{.Status}}' || true)"
    if [[ -z "$docker_status" ]]; then
      echo "litellm-dev is not running; starting it now..."
      docker compose -f docker-compose.dev.yml up -d litellm-dev
      echo "Waiting briefly for litellm-dev to finish booting..."
      sleep 10
    fi
  fi
else
  echo "Acceptance harness targeting $TARGET_LITELLM_BASE_URL; skipping litellm-dev lifecycle management."
fi

exec "$PYTHON_BIN" scripts/local-ci/run_acceptance.py \
  --config "$CONFIG_PATH" \
  --claude-fanout-mode "$CLAUDE_FANOUT_MODE" \
  --write-artifact "$ARTIFACT_PATH"
