#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_PATH="${1:-$ROOT/.analysis/artifacts/local-acceptance-$(date -u +%Y%m%dT%H%M%SZ).json}"
REBUILD_LITELLM_DEV="${REBUILD_LITELLM_DEV:-1}"
BUILD_STATE_PATH="${BUILD_STATE_PATH:-$ROOT/.analysis/artifacts/litellm-dev-build-state.json}"

cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export LANGFUSE_QUERY_URL="${LANGFUSE_QUERY_URL:-http://127.0.0.1:3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLAUDE_FANOUT_MODE="${CLAUDE_FANOUT_MODE:-minimal}"

compute_build_fingerprint() {
  "$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
exclude_prefixes = [
    ".git/",
    ".analysis/",
    "captures/",
    "scripts/local-ci/",
    ".gemini/",
    ".codex/",
]
exclude_exact = {
    "langfuse-traces.png",
    "litellm/integrations/aawm_agent_identity.py",
    "litellm/integrations/aawm_payload_capture.py",
    "litellm-dev-config.yaml",
}

def include(path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in exclude_exact:
        return False
    for prefix in exclude_prefixes:
        if rel.startswith(prefix):
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
  local state_json current_fingerprint previous_fingerprint image_present container_present
  state_json="$(compute_build_fingerprint)"
  current_fingerprint="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read())["fingerprint"])' <<<"$state_json")"

  previous_fingerprint=""
  if [[ -f "$BUILD_STATE_PATH" ]]; then
    previous_fingerprint="$("$PYTHON_BIN" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); print(json.loads(p.read_text()).get("fingerprint",""))' "$BUILD_STATE_PATH" 2>/dev/null || true)"
  fi

  image_present="$(docker images -q litellm-litellm-dev 2>/dev/null || true)"
  container_present="$(docker ps -a --filter name=^litellm-dev$ --format '{{.ID}}' 2>/dev/null || true)"

  if [[ -z "$image_present" || -z "$container_present" ]]; then
    echo "$state_json" > "$BUILD_STATE_PATH"
    return 0
  fi

  if [[ "$current_fingerprint" != "$previous_fingerprint" ]]; then
    echo "$state_json" > "$BUILD_STATE_PATH"
    return 0
  fi

  return 1
}

if [[ "$REBUILD_LITELLM_DEV" == "1" ]] && should_rebuild_litellm_dev; then
  echo "litellm-dev image inputs changed; rebuilding and recreating the container..."
  docker compose -f docker-compose.dev.yml build litellm-dev
  docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev
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

exec "$PYTHON_BIN" scripts/local-ci/run_acceptance.py \
  --config scripts/local-ci/config.json \
  --claude-fanout-mode "$CLAUDE_FANOUT_MODE" \
  --write-artifact "$ARTIFACT_PATH"
