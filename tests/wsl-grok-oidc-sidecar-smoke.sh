#!/usr/bin/env bash
# Static smoke checks for the WSL-local Grok OIDC single-writer sidecar.
# Validates compose invariants and launcher safety without mutating Docker.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${repo_root}/docker-compose.wsl-grok-oidc.yml"
launcher="${repo_root}/scripts/ensure-wsl-grok-oidc-sidecar.sh"
docs_file="${repo_root}/docs/aawm-provider-status-observations.md"

pass_count=0
fail_count=0

pass() {
  echo "PASS: $*"
  pass_count=$((pass_count + 1))
}

fail() {
  echo "FAIL: $*" >&2
  fail_count=$((fail_count + 1))
}

require_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    pass "file exists: ${path#"$repo_root"/}"
  else
    fail "missing file: ${path#"$repo_root"/}"
  fi
}

require_executable() {
  local path="$1"
  if [[ -x "$path" ]]; then
    pass "executable: ${path#"$repo_root"/}"
  else
    fail "not executable: ${path#"$repo_root"/}"
  fi
}

require_contains() {
  local path="$1"
  local pattern="$2"
  local label="$3"
  if grep -Eq -- "$pattern" "$path"; then
    pass "$label"
  else
    fail "$label (pattern not found: $pattern)"
  fi
}

require_absent() {
  local path="$1"
  local pattern="$2"
  local label="$3"
  if grep -Eq -- "$pattern" "$path"; then
    fail "$label (forbidden pattern present: $pattern)"
  else
    pass "$label"
  fi
}

count_services() {
  python3 - "$compose_file" <<'INNER'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
lines = text.splitlines()
in_services = False
services = []
for line in lines:
    if line.startswith("services:"):
        in_services = True
        continue
    if in_services:
        if line and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("#"):
            break
        if line.startswith("  ") and not line.startswith("   ") and line.strip().endswith(":"):
            name = line.strip().rstrip(":")
            if name and not name.startswith("#"):
                services.append(name)
print(len(services))
print(",".join(services))
INNER
}

echo "== XAI-003 WSL Grok OIDC sidecar smoke (static) =="

require_file "$compose_file"
require_file "$launcher"
require_file "$docs_file"
require_executable "$launcher"

if bash -n "$launcher"; then
  pass "bash -n launcher"
else
  fail "bash -n launcher"
fi

if bash -n "$0"; then
  pass "bash -n smoke"
else
  fail "bash -n smoke"
fi

service_info="$(count_services)"
service_count="$(printf '%s\n' "$service_info" | sed -n '1p')"
service_names="$(printf '%s\n' "$service_info" | sed -n '2p')"
if [[ "$service_count" == "1" && "$service_names" == "wsl-grok-oidc-refresh" ]]; then
  pass "compose defines exactly one service: wsl-grok-oidc-refresh"
else
  fail "compose must define exactly one service wsl-grok-oidc-refresh (found count=${service_count} names=${service_names})"
fi

require_contains "$compose_file" 'container_name:[[:space:]]*aawm-wsl-grok-oidc-refresh' \
  "compose container_name is aawm-wsl-grok-oidc-refresh"
require_contains "$compose_file" 'image:[[:space:]]*aawm-provider-status-observations:prod' \
  "compose reuses aawm-provider-status-observations:prod"
require_contains "$compose_file" 'restart:[[:space:]]*unless-stopped' \
  "compose restart policy is unless-stopped"
require_contains "$compose_file" 'no-new-privileges:true' \
  "compose sets no-new-privileges"
require_contains "$compose_file" '/home/zepfu/\.grok:/home/zepfu/\.grok' \
  "compose mounts /home/zepfu/.grok"
require_absent "$compose_file" '/home/zepfu/\.codex|/home/zepfu/\.litellm/xai|/home/zepfu/\.kimi-code|/home/zepfu/\.alibaba' \
  "compose does not mount other credential directories"
require_absent "$compose_file" 'aawm-litellm|litellm-dev' \
  "compose does not reference LiteLLM proxy services"
require_absent "$compose_file" 'build:|depends_on:' \
  "compose has no build or depends_on"

require_contains "$compose_file" 'AAWM_GROK_OIDC_REFRESH_ENABLED=1' \
  "Grok OIDC refresh enabled"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE=/home/zepfu/\.grok/auth\.json' \
  "Grok auth file path set"
require_contains "$compose_file" 'AAWM_GROK_OIDC_LOCK_FILE=/home/zepfu/\.grok/auth\.json\.lock' \
  "Grok lock file path set"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE_UID=1000' \
  "Grok auth uid 1000"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE_GID=1000' \
  "Grok auth gid 1000"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE_MODE=0o600' \
  "Grok auth mode 0o600"
require_contains "$compose_file" 'AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=300' \
  "Grok refresh interval 300"
require_contains "$compose_file" 'AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS=900' \
  "Grok refresh buffer 900"
require_contains "$compose_file" 'AAWM_GROK_OIDC_FORCE_REFRESH=0' \
  "Grok force refresh disabled"
require_contains "$compose_file" 'AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS=30' \
  "Grok HTTP timeout 30"

for flag in \
  'AAWM_CODEX_OAUTH_REFRESH_ENABLED=0' \
  'AAWM_XAI_OAUTH_REFRESH_ENABLED=0' \
  'AAWM_KIMI_OAUTH_REFRESH_ENABLED=0' \
  'AAWM_KIMI_USAGE_POLL_ENABLED=0' \
  'AAWM_ALIBABA_QUOTA_POLL_ENABLED=0' \
  'AAWM_CODEX_RESET_CREDIT_POLL_ENABLED=0' \
  'AAWM_GROK_BILLING_POLL_ENABLED=0' \
  'AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED=0' \
  'AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED=0'
do
  require_contains "$compose_file" "$flag" "disabled unrelated task: $flag"
done

require_contains "$compose_file" 'AAWM_PROVIDER_STATUS_APPLY=0' \
  "provider-status apply disabled"
require_contains "$compose_file" 'AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START=0' \
  "schema setup disabled"
require_contains "$compose_file" 'AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER=0' \
  "pgbouncer not required"

require_contains "$compose_file" 'healthcheck:' \
  "healthcheck defined"
require_contains "$compose_file" 'remaining > 600' \
  "healthcheck fails before 300s rejection boundary"
require_contains "$compose_file" "record.get\\('refresh_token'\\)" \
  "healthcheck requires a refresh token"
require_contains "$compose_file" "record.get\\('oidc_issuer'\\)" \
  "healthcheck validates the Grok OIDC issuer"
require_contains "$compose_file" "record.get\\('oidc_client_id'\\)" \
  "healthcheck validates the Grok OIDC client id"

python3 - "$compose_file" <<'INNER'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
hc = text[text.index("healthcheck:") :]
if re.search(r"print\(|echo ", hc):
    raise SystemExit("healthcheck must not print or echo")
if "remaining > 600" not in hc:
    raise SystemExit("healthcheck missing remaining > 600 gate")
if re.search(r"Bearer |Authorization:|client_secret\s*=", text):
    raise SystemExit("compose has credential header/secret assignment literals")
if "json.dumps(payload" not in text:
    raise SystemExit("command loop missing sanitized JSON emit path")
if "import scripts.grok_oidc_refresh as grok" not in text:
    raise SystemExit("compose command must use scripts.grok_oidc_refresh")
if re.search(
    r"run_provider_status_observations_loop|DEFAULT_ENDPOINTS|collect_observations",
    text,
):
    raise SystemExit("compose must not invoke multi-provider observation loop")
if "entrypoint:" not in text:
    raise SystemExit("compose must override entrypoint for Grok-only loop")
print("secret_safe_and_grok_only_ok")
INNER
pass "healthcheck/command are secret-safe and Grok-only"

require_contains "$launcher" '--status' "launcher supports --status"
require_contains "$launcher" '--apply' "launcher supports --apply"
require_contains "$launcher" '--stop' "launcher supports --stop"
require_contains "$launcher" 'mode="status"' "launcher defaults to status"
require_contains "$launcher" 'compose up -d --no-deps --no-build' \
  "apply uses --no-deps --no-build"
require_contains "$launcher" 'compose stop' \
  "stop uses compose stop"
require_contains "$launcher" 'capture_proxy_snapshots' \
  "launcher captures proxy snapshots"
require_contains "$launcher" 'assert_proxy_snapshots_unchanged' \
  "launcher asserts proxy snapshots unchanged"
require_contains "$launcher" 'assert_proxy_snapshots_present' \
  "launcher requires both proxy baselines"
require_contains "$launcher" 'wait_for_sidecar_healthy' \
  "launcher waits for sidecar health"
require_contains "$launcher" 'aawm-litellm' \
  "launcher tracks aawm-litellm snapshot"
require_contains "$launcher" 'litellm-dev' \
  "launcher tracks litellm-dev snapshot"
require_contains "$launcher" 'preflight_image' \
  "launcher preflights image"
require_contains "$launcher" 'preflight_credential' \
  "launcher preflights credential"

service_ref="\$service_name"
while IFS= read -r line; do
  if [[ "$line" =~ compose\ up ]]; then
    if [[ "$line" == *"--no-deps"* && "$line" == *"--no-build"* && "$line" == *"$service_ref"* ]]; then
      pass "compose up invocation is service-scoped with --no-deps --no-build"
    else
      fail "unsafe compose up: $line"
    fi
  fi
  if [[ "$line" =~ compose\ down ]]; then
    fail "launcher must not invoke compose down: $line"
  fi
done < <(grep -n 'compose ' "$launcher" || true)

if grep -E 'compose (up|stop|rm|start|restart|kill)' "$launcher" | grep -E 'aawm-litellm|litellm-dev' >/dev/null; then
  fail "mutating compose command mentions a LiteLLM proxy service"
else
  pass "mutating compose commands do not mention proxy services"
fi

help_out="$("$launcher" --help 2>&1 || true)"
if printf '%s\n' "$help_out" | grep -Eq -- '--status|--apply|--stop'; then
  pass "launcher --help documents modes"
else
  fail "launcher --help missing mode docs"
fi

if "$launcher" --explode >/dev/null 2>&1; then
  fail "launcher accepted unknown argument"
else
  pass "launcher rejects unknown arguments"
fi

require_contains "$docs_file" 'WSL-local single writer|WSL-local single-writer|WSL host' \
  "docs mention WSL-local writer ownership"
require_contains "$docs_file" 'break-glass|manual Grok login' \
  "docs state manual Grok login is break-glass"
require_contains "$docs_file" 'read-only' \
  "docs state LiteLLM consumers are read-only"
require_contains "$docs_file" 'oa_xai|managed xAI OAuth' \
  "docs separate managed oa_xai OAuth"
require_contains "$docs_file" 'ensure-wsl-grok-oidc-sidecar|docker-compose.wsl-grok-oidc' \
  "docs reference the WSL sidecar artifacts"
require_contains "$docs_file" 'no restart|without restart|require no restart' \
  "docs state consumers require no restart"

echo
echo "summary: pass=${pass_count} fail=${fail_count}"
if [[ "$fail_count" -ne 0 ]]; then
  exit 1
fi
echo "smoke_ok"
