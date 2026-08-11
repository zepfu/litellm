#!/usr/bin/env bash
# Static smoke checks for the WSL-local dual-writer Grok OIDC + managed xAI
# OAuth sidecar. Validates compose invariants and launcher safety without
# mutating Docker.
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

echo "== XAI-004 WSL dual Grok OIDC + managed xAI OAuth sidecar smoke (static) =="

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
require_contains "$compose_file" '/home/zepfu/\.litellm/xai:/home/zepfu/\.litellm/xai' \
  "compose mounts /home/zepfu/.litellm/xai"
require_absent "$compose_file" '/home/zepfu/\.codex|/home/zepfu/\.kimi-code|/home/zepfu/\.alibaba' \
  "compose does not mount Codex/Kimi/Alibaba credential directories"
require_absent "$compose_file" 'aawm-litellm|litellm-dev' \
  "compose does not reference LiteLLM proxy services"
require_absent "$compose_file" 'build:|depends_on:' \
  "compose has no build or depends_on"

require_contains "$compose_file" 'AAWM_GROK_OIDC_REFRESH_ENABLED=1' \
  "Grok OIDC refresh enabled"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_REFRESH_ENABLED=1' \
  "managed xAI OAuth refresh enabled"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE=/home/zepfu/\.grok/auth\.json' \
  "Grok auth file path set"
require_contains "$compose_file" 'AAWM_GROK_OIDC_LOCK_FILE=/home/zepfu/\.grok/auth\.json\.lock' \
  "Grok lock file path set"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_AUTH_FILE=/home/zepfu/\.litellm/xai/oauth-auth\.json' \
  "managed xAI auth file path set"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_LOCK_FILE=/home/zepfu/\.litellm/xai/oauth-auth\.json\.lock' \
  "managed xAI lock file path set"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE_UID=1000' \
  "Grok auth uid 1000"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE_GID=1000' \
  "Grok auth gid 1000"
require_contains "$compose_file" 'AAWM_GROK_OIDC_AUTH_FILE_MODE=0o600' \
  "Grok auth mode 0o600"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_AUTH_FILE_UID=0' \
  "managed xAI auth uid 0"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_AUTH_FILE_GID=0' \
  "managed xAI auth gid 0"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_AUTH_FILE_MODE=0o600' \
  "managed xAI auth mode 0o600"
require_contains "$compose_file" 'AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=300' \
  "Grok refresh interval 300"
require_contains "$compose_file" 'AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS=900' \
  "Grok refresh buffer 900"
require_contains "$compose_file" 'AAWM_GROK_OIDC_FORCE_REFRESH=0' \
  "Grok force refresh disabled"
require_contains "$compose_file" 'AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS=30' \
  "Grok HTTP timeout 30"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS=300' \
  "managed xAI refresh interval 300"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS=900' \
  "managed xAI refresh buffer 900"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_FORCE_REFRESH=0' \
  "managed xAI force refresh disabled"
require_contains "$compose_file" 'AAWM_XAI_OAUTH_HTTP_TIMEOUT_SECONDS=30' \
  "managed xAI HTTP timeout 30"

for flag in \
  'AAWM_CODEX_OAUTH_REFRESH_ENABLED=0' \
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
require_contains "$compose_file" '^[[:space:]]+- CMD$' \
  "healthcheck uses exec-form CMD"
require_absent "$compose_file" 'CMD-SHELL' \
  "healthcheck does not embed multiline Python in CMD-SHELL"
require_contains "$compose_file" 'remaining.* > 600|return .* > 600|total_seconds\(\) > 600' \
  "healthcheck fails before 300s rejection boundary"
require_contains "$compose_file" "record.get\\('refresh_token'\\)|refresh_token" \
  "healthcheck requires a refresh token"
require_contains "$compose_file" "oidc_issuer|require_issuer" \
  "healthcheck validates native issuer requirement"
require_contains "$compose_file" "oidc_client_id|client_id" \
  "healthcheck validates client id"
require_contains "$compose_file" 'oauth-auth\.json' \
  "healthcheck covers managed oauth-auth.json"
require_contains "$compose_file" 'auth\.json' \
  "healthcheck covers native auth.json"

python3 - "$compose_file" <<'INNER'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
hc = text[text.index("healthcheck:") :]
if re.search(r"print\(|echo ", hc):
    raise SystemExit("healthcheck must not print or echo")
if "total_seconds() > 600" not in hc and "remaining > 600" not in hc:
    raise SystemExit("healthcheck missing remaining > 600 gate")
if "/home/zepfu/.grok/auth.json" not in hc:
    raise SystemExit("healthcheck must validate native auth.json")
if "/home/zepfu/.litellm/xai/oauth-auth.json" not in hc:
    raise SystemExit("healthcheck must validate managed oauth-auth.json")
if "require_issuer" not in hc and "oidc_issuer" not in hc:
    raise SystemExit("healthcheck missing issuer handling")
if re.search(r"Bearer |Authorization:|client_secret\s*=", text):
    raise SystemExit("compose has credential header/secret assignment literals")
if "json.dumps(payload" not in text:
    raise SystemExit("command loop missing sanitized JSON emit path")
if "import scripts.grok_oidc_refresh as grok" not in text:
    raise SystemExit("compose command must use scripts.grok_oidc_refresh")
if "import scripts.xai_oauth_refresh as xai" not in text:
    raise SystemExit("compose command must use scripts.xai_oauth_refresh")
if "event\": \"grok_oidc_refresh\"" not in text and "event\": 'grok_oidc_refresh'" not in text:
    # YAML embeds python with double quotes
    if '"event": "grok_oidc_refresh"' not in text:
        raise SystemExit("compose must emit grok_oidc_refresh events")
if '"event": "xai_oauth_refresh"' not in text:
    raise SystemExit("compose must emit xai_oauth_refresh events")
if '"event": "xai_oauth_metadata_repair"' not in text:
    raise SystemExit("compose must emit xai_oauth_metadata_repair events")
if '"event": "grok_oidc_metadata_repair"' not in text:
    raise SystemExit("compose must emit grok_oidc_metadata_repair events")
if re.search(
    r"run_provider_status_observations_loop|DEFAULT_ENDPOINTS|collect_observations",
    text,
):
    raise SystemExit("compose must not invoke multi-provider observation loop")
if "entrypoint:" not in text:
    raise SystemExit("compose must override entrypoint for dual credential loop")
# Only the two credential mounts.
mounts = re.findall(r"^\s*-\s*(/home/zepfu/[^\s:]+):/home/zepfu/", text, re.M)
if sorted(mounts) != sorted(["/home/zepfu/.grok", "/home/zepfu/.litellm/xai"]):
    raise SystemExit(f"unexpected volume mounts: {mounts}")
print("secret_safe_and_dual_credential_ok")
INNER
pass "healthcheck/command are secret-safe and dual-credential"

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
require_contains "$launcher" 'preflight_native_credential|preflight_credentials' \
  "launcher preflights native credential"
require_contains "$launcher" 'preflight_managed_credential|preflight_credentials' \
  "launcher preflights managed credential"
require_contains "$launcher" 'read_managed_credential_via_docker|docker run --rm --read-only' \
  "launcher can validate unreadable managed credential via read-only docker"
require_contains "$launcher" '--entrypoint python' \
  "launcher overrides the provider-status image entrypoint for managed preflight"
require_contains "$launcher" 'python3 -c' \
  "managed credential validator preserves stdin for piped JSON"
require_absent "$launcher" 'local raw|raw="\$\(read_managed_credential_via_docker' \
  "launcher does not retain managed credential JSON in a shell variable"
require_contains "$launcher" 'native_auth_file|managed_auth_file' \
  "launcher tracks both credential paths"

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

require_contains "$docs_file" 'WSL-local|WSL host' \
  "docs mention WSL-local writer ownership"
require_contains "$docs_file" 'break-glass|manual Grok login' \
  "docs state manual Grok login is break-glass"
require_contains "$docs_file" 'read-only' \
  "docs state LiteLLM consumers are read-only"
require_contains "$docs_file" 'oa_xai|managed xAI OAuth' \
  "docs cover managed oa_xai OAuth"
require_contains "$docs_file" 'ensure-wsl-grok-oidc-sidecar|docker-compose.wsl-grok-oidc' \
  "docs reference the WSL sidecar artifacts"
require_contains "$docs_file" 'no restart|without restart|require no restart' \
  "docs state consumers require no restart"
require_contains "$docs_file" 'XAI-004|dual|managed xAI OAuth' \
  "docs cover dual credential ownership (XAI-004)"
require_contains "$docs_file" 'oauth-auth\.json' \
  "docs name managed oauth-auth.json path"
require_contains "$docs_file" 'xai_oauth_refresh' \
  "docs name managed xai_oauth_refresh events/task"

echo
guard_probe_dir="$(mktemp -d)"
trap 'rm -rf "$guard_probe_dir"' EXIT
fake_bin_dir="${guard_probe_dir}/bin"
fake_creds_dir="${guard_probe_dir}/creds"
mkdir -p "$fake_bin_dir" "$fake_creds_dir"
real_stat="$(command -v stat)"

cat >"${fake_bin_dir}/docker" <<'FAKE_DOCKER'
#!/usr/bin/env bash
echo "docker $*" >>"$FAKE_DOCKER_CALLS"
case "${1:-}" in
  image) exit 0 ;;
  compose)
    [[ " $* " == *" stop "* ]] && printf 'exited\n' >"$FAKE_DOCKER_STATE"
    [[ " $* " == *" up "* ]] && printf 'running\n' >"$FAKE_DOCKER_STATE"
    exit 0
    ;;
  inspect)
    template="${3:-}"
    target="${@: -1}"
    case "$template" in
      '{{.Id}}') printf 'id-%s\n' "$target" ;;
      '{{.State.StartedAt}}') printf '2026-08-11T00:00:00Z\n' ;;
      '{{.RestartCount}}') printf '0\n' ;;
      '{{.State.Status}}')
        if [[ "$target" == "aawm-wsl-grok-oidc-refresh" ]]; then
          cat "$FAKE_DOCKER_STATE"
        else
          printf 'running\n'
        fi
        ;;
      *'.State.Health'*) printf 'healthy\n' ;;
      '{{.Config.Image}}') printf 'aawm-provider-status-observations:prod\n' ;;
    esac
    exit 0
    ;;
  *) exit 0 ;;
esac
FAKE_DOCKER
chmod +x "${fake_bin_dir}/docker"

cat >"${fake_bin_dir}/stat" <<'FAKE_STAT'
#!/usr/bin/env bash
if [[ "${1:-}" == "-c" && "$#" -eq 3 ]]; then
  case "${3}:${2}" in
    "${FAKE_NATIVE_AUTH_FILE}:"*|"${FAKE_MANAGED_AUTH_FILE}:"*)
      echo "stat $*" >>"$FAKE_STAT_CALLS"
      [[ "${FAKE_STAT_REJECT_AUTH:-0}" == "1" ]] && exit 1
      ;;
  esac
  case "${3}:${2}" in
    "${FAKE_NATIVE_AUTH_FILE}:%a"|"${FAKE_MANAGED_AUTH_FILE}:%a") printf '600\n'; exit 0 ;;
    "${FAKE_NATIVE_AUTH_FILE}:%u"|"${FAKE_NATIVE_AUTH_FILE}:%g") printf '1000\n'; exit 0 ;;
    "${FAKE_MANAGED_AUTH_FILE}:%u"|"${FAKE_MANAGED_AUTH_FILE}:%g") printf '0\n'; exit 0 ;;
  esac
fi
exec "$REAL_STAT" "$@"
FAKE_STAT
chmod +x "${fake_bin_dir}/stat"

printf '%s' '{"https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {"oidc_issuer": "https://auth.x.ai", "oidc_client_id": "b1a00492-073a-47ea-816f-4c329264a828", "expires_at": 9999999999, "refresh_token": "test-only", "key": "test-only"}}' \
  >"${fake_creds_dir}/auth.json"
printf '%s' '{"client_id": "b1a00492-073a-47ea-816f-4c329264a828", "expires_at": 9999999999, "refresh_token": "test-only", "key": "test-only"}' \
  >"${fake_creds_dir}/oauth-auth.json"
chmod 600 "${fake_creds_dir}/auth.json" "${fake_creds_dir}/oauth-auth.json"

run_fixture() {
  local osrelease_fixture="$1" mode="$2" label="$3" reject_auth_stat="${4:-0}"
  probe_calls="${guard_probe_dir}/calls-${label}.log"
  probe_stat_calls="${guard_probe_dir}/stat-calls-${label}.log"
  probe_state="${guard_probe_dir}/state-${label}"
  : >"$probe_calls"
  : >"$probe_stat_calls"
  printf 'running\n' >"$probe_state"
  probe_rc=0
  probe_output="$(
    PATH="${fake_bin_dir}:${PATH}" \
    REAL_STAT="$real_stat" \
    FAKE_DOCKER_CALLS="$probe_calls" \
    FAKE_DOCKER_STATE="$probe_state" \
    FAKE_STAT_CALLS="$probe_stat_calls" \
    FAKE_STAT_REJECT_AUTH="$reject_auth_stat" \
    FAKE_NATIVE_AUTH_FILE="${fake_creds_dir}/auth.json" \
    FAKE_MANAGED_AUTH_FILE="${fake_creds_dir}/oauth-auth.json" \
    WSL_GROK_OIDC_OSRELEASE_FILE="$osrelease_fixture" \
    WSL_GROK_OIDC_DOCKER_BIN="${fake_bin_dir}/docker" \
    WSL_GROK_OIDC_COMPOSE_FILE="$compose_file" \
    WSL_GROK_OIDC_AUTH_FILE="${fake_creds_dir}/auth.json" \
    WSL_XAI_OAUTH_AUTH_FILE="${fake_creds_dir}/oauth-auth.json" \
    WSL_GROK_OIDC_HEALTH_TIMEOUT_SECONDS=5 \
    "$launcher" "$mode" 2>&1
  )" || probe_rc=$?
}

assert_refused_apply() {
  local osrelease_fixture="$1" label="$2"
  run_fixture "$osrelease_fixture" --apply "$label"
  if [[ "$probe_rc" -eq 1 ]] && grep -q 'apply_refused_non_wsl_host' <<<"$probe_output"; then
    pass "${label}: apply refused"
  else
    fail "${label}: expected refusal, rc=${probe_rc}; output: ${probe_output}"
  fi
  if [[ ! -s "$probe_calls" ]]; then
    pass "${label}: no Docker calls"
  else
    fail "${label}: Docker was called: $(cat "$probe_calls")"
  fi
}

printf '5.15.0-105-generic\n' >"${guard_probe_dir}/osrelease-nonwsl"
printf '5.15.153.1-microsoft-standard-WSL2\n' >"${guard_probe_dir}/osrelease-wsl"

assert_refused_apply "${guard_probe_dir}/missing-osrelease" "unreadable-osrelease"
assert_refused_apply "${guard_probe_dir}/osrelease-nonwsl" "marker-free-osrelease"

run_fixture "${guard_probe_dir}/osrelease-nonwsl" --status "nonwsl-status" 1
if [[ "$probe_rc" -eq 0 ]] \
  && grep -q 'credential_preflight=skipped_non_wsl' <<<"$probe_output" \
  && grep -q 'status_ok' <<<"$probe_output" \
  && [[ ! -s "$probe_stat_calls" ]]; then
  pass "non-WSL --status skips credential preflight"
else
  fail "non-WSL --status used credential preflight, rc=${probe_rc}; output: ${probe_output}"
fi

run_fixture "${guard_probe_dir}/osrelease-nonwsl" --stop "nonwsl-stop"
if [[ "$probe_rc" -eq 0 ]] && grep -q 'stop_ok' <<<"$probe_output"; then
  pass "non-WSL --stop remains available"
else
  fail "non-WSL --stop failed, rc=${probe_rc}; output: ${probe_output}"
fi

run_fixture "${guard_probe_dir}/osrelease-wsl" --apply "wsl-apply"
expected_compose_call="docker compose -f ${compose_file} up -d --no-deps --no-build wsl-grok-oidc-refresh"
compose_calls="$(grep '^docker compose ' "$probe_calls" || true)"
if [[ "$probe_rc" -eq 0 ]] && grep -q 'apply_ok service=wsl-grok-oidc-refresh proxies_unchanged=true' <<<"$probe_output"; then
  pass "WSL apply preserves proxy snapshots"
else
  fail "WSL apply failed, rc=${probe_rc}; output: ${probe_output}"
fi
if [[ "$compose_calls" == "$expected_compose_call" ]]; then
  pass "WSL apply uses exactly the service-scoped compose up"
else
  fail "unexpected WSL apply compose calls: ${compose_calls:-none}"
fi

echo "summary: pass=${pass_count} fail=${fail_count}"
if [[ "$fail_count" -ne 0 ]]; then
  exit 1
fi
echo "smoke_ok"
