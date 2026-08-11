#!/usr/bin/env bash
# Ensure the WSL-local dual-writer Grok OIDC + managed xAI OAuth refresh sidecar.
#
# Default: --status
# Mutating modes start/stop only the dedicated service from
# docker-compose.wsl-grok-oidc.yml and refuse to touch aawm-litellm or
# litellm-dev. Proxy identity (container ID, start timestamp, restart count)
# is snapshotted before and after every mutation.
# --apply is WSL-only; --status and --stop remain available elsewhere.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${WSL_GROK_OIDC_COMPOSE_FILE:-${repo_root}/docker-compose.wsl-grok-oidc.yml}"
service_name="${WSL_GROK_OIDC_SERVICE_NAME:-wsl-grok-oidc-refresh}"
container_name="${WSL_GROK_OIDC_CONTAINER_NAME:-aawm-wsl-grok-oidc-refresh}"
image_name="${WSL_GROK_OIDC_IMAGE:-aawm-provider-status-observations:prod}"
native_auth_file="${WSL_GROK_OIDC_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
managed_auth_file="${WSL_XAI_OAUTH_AUTH_FILE:-/home/zepfu/.litellm/xai/oauth-auth.json}"
docker_bin="${WSL_GROK_OIDC_DOCKER_BIN:-docker}"
osrelease_file="${WSL_GROK_OIDC_OSRELEASE_FILE:-/proc/sys/kernel/osrelease}"

declare -a proxy_containers=(aawm-litellm litellm-dev)

mode="status"
for arg in "$@"; do
  case "$arg" in
    --status) mode="status" ;;
    --apply) mode="apply" ;;
    --stop) mode="stop" ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  scripts/ensure-wsl-grok-oidc-sidecar.sh [--status|--apply|--stop]

  --status  Preflight image + native/managed credentials, report sidecar +
            proxy snapshots (default). Status works before activation.
  --apply   Start/recreate only the dedicated dual-credential service with
            --no-deps --no-build after proving both LiteLLM proxies stay
            unchanged. WSL-only: refused fail-closed on a non-WSL host.
  --stop    Stop only the dedicated service with --no-deps, then prove both
            LiteLLM proxies stayed unchanged.
USAGE
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg" >&2
      echo "use --status, --apply, or --stop" >&2
      exit 2
      ;;
  esac
done

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "$*"
}

detect_host_kind() {
  if grep -qiE 'microsoft|wsl' "$osrelease_file" 2>/dev/null; then
    printf 'wsl\n'
  else
    printf 'non-wsl\n'
  fi
}

require_wsl_host_for_apply() {
  local host_kind
  host_kind="$(detect_host_kind)"
  [[ "$host_kind" == "wsl" ]] \
    || die "apply_refused_non_wsl_host: --apply may only start this WSL-only sidecar on a WSL host (host_kind=${host_kind}); use --status or --stop to inspect/stop an accidentally present container"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

compose() {
  # Intentionally never accepts a bare up/down for the whole project.
  "$docker_bin" compose -f "$compose_file" "$@"
}

proxy_snapshot_line() {
  local container="$1"
  local id started restarts
  if ! id="$("$docker_bin" inspect -f '{{.Id}}' "$container" 2>/dev/null)"; then
    printf '%s|missing||\n' "$container"
    return 0
  fi
  started="$("$docker_bin" inspect -f '{{.State.StartedAt}}' "$container")"
  restarts="$("$docker_bin" inspect -f '{{.RestartCount}}' "$container")"
  printf '%s|%s|%s|%s\n' "$container" "$id" "$started" "$restarts"
}

capture_proxy_snapshots() {
  local container
  for container in "${proxy_containers[@]}"; do
    proxy_snapshot_line "$container"
  done
}

assert_proxy_snapshots_unchanged() {
  local before_file="$1"
  local after_file="$2"
  local before_line after_line container
  while IFS= read -r before_line; do
    container="${before_line%%|*}"
    after_line="$(grep -E "^${container}\\|" "$after_file" || true)"
    [[ -n "$after_line" ]] || die "missing post-mutation snapshot for ${container}"
    if [[ "$before_line" != "$after_line" ]]; then
      die "LiteLLM proxy snapshot changed for ${container}: before=${before_line} after=${after_line}"
    fi
  done <"$before_file"
}

assert_proxy_snapshots_present() {
  local snapshot_file="$1"
  local line container id _started _restarts
  while IFS= read -r line; do
    IFS='|' read -r container id _started _restarts <<<"$line"
    [[ "$id" != "missing" && -n "$id" ]] \
      || die "required LiteLLM proxy is not running: ${container}"
  done <"$snapshot_file"
}

wait_for_sidecar_healthy() {
  local timeout_seconds="${WSL_GROK_OIDC_HEALTH_TIMEOUT_SECONDS:-120}"
  local deadline=$((SECONDS + timeout_seconds))
  local state health
  while ((SECONDS < deadline)); do
    if "$docker_bin" inspect "$container_name" >/dev/null 2>&1; then
      state="$("$docker_bin" inspect -f '{{.State.Status}}' "$container_name")"
      health="$("$docker_bin" inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container_name")"
      [[ "$state" == "running" ]] \
        || die "sidecar ${container_name} entered state=${state} before becoming healthy"
      if [[ "$health" == "healthy" ]]; then
        return 0
      fi
      [[ "$health" != "unhealthy" ]] \
        || die "sidecar ${container_name} became unhealthy during startup"
    fi
    sleep 2
  done
  die "sidecar ${container_name} did not become healthy within ${timeout_seconds}s"
}

assert_compose_contract() {
  [[ -f "$compose_file" ]] || die "compose file missing: $compose_file"
  local text
  text="$(cat "$compose_file")"
  grep -Eq 'container_name:[[:space:]]*aawm-wsl-grok-oidc-refresh' <<<"$text" \
    || die "compose must name container aawm-wsl-grok-oidc-refresh"
  grep -Eq 'image:[[:space:]]*aawm-provider-status-observations:prod' <<<"$text" \
    || die "compose must reuse image aawm-provider-status-observations:prod"
  grep -Eq 'restart:[[:space:]]*unless-stopped' <<<"$text" \
    || die "compose must set restart: unless-stopped"
  grep -Fq 'no-new-privileges:true' <<<"$text" \
    || die "compose must set no-new-privileges:true"
  grep -Fq '/home/zepfu/.grok:/home/zepfu/.grok' <<<"$text" \
    || die "compose must mount /home/zepfu/.grok"
  grep -Fq '/home/zepfu/.litellm/xai:/home/zepfu/.litellm/xai' <<<"$text" \
    || die "compose must mount /home/zepfu/.litellm/xai"
  # Refuse accidental multi-service or proxy coupling in this file.
  if grep -Eq 'aawm-litellm|litellm-dev' <<<"$text"; then
    die "compose file must not reference aawm-litellm or litellm-dev"
  fi
  if grep -Eq '/home/zepfu/\.codex|/home/zepfu/\.kimi-code|/home/zepfu/\.alibaba' <<<"$text"; then
    die "compose must not mount Codex, Kimi, or Alibaba credential dirs"
  fi
  grep -Fq 'AAWM_GROK_OIDC_REFRESH_ENABLED=1' <<<"$text" \
    || die "compose must enable Grok OIDC refresh"
  grep -Fq 'AAWM_XAI_OAUTH_REFRESH_ENABLED=1' <<<"$text" \
    || die "compose must enable managed xAI OAuth refresh"
  grep -Fq 'AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=300' <<<"$text" \
    || die "compose must set Grok refresh interval 300"
  grep -Fq 'AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS=900' <<<"$text" \
    || die "compose must set Grok refresh buffer 900"
  grep -Fq 'AAWM_GROK_OIDC_FORCE_REFRESH=0' <<<"$text" \
    || die "compose must set Grok force refresh 0"
  grep -Fq 'AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS=300' <<<"$text" \
    || die "compose must set managed xAI refresh interval 300"
  grep -Fq 'AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS=900' <<<"$text" \
    || die "compose must set managed xAI refresh buffer 900"
  grep -Fq 'AAWM_XAI_OAUTH_FORCE_REFRESH=0' <<<"$text" \
    || die "compose must set managed xAI force refresh 0"
  grep -Fq 'AAWM_PROVIDER_STATUS_APPLY=0' <<<"$text" \
    || die "compose must disable provider-status DB apply"
  grep -Fq 'AAWM_CODEX_OAUTH_REFRESH_ENABLED=0' <<<"$text" \
    || die "compose must disable Codex OAuth refresh"
  grep -Fq 'AAWM_KIMI_OAUTH_REFRESH_ENABLED=0' <<<"$text" \
    || die "compose must disable Kimi OAuth refresh"
  grep -Fq 'AAWM_GROK_BILLING_POLL_ENABLED=0' <<<"$text" \
    || die "compose must disable Grok billing poll"
  grep -Fq 'AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED=0' <<<"$text" \
    || die "compose must disable anomaly scan"
  grep -Fq 'scripts.grok_oidc_refresh' <<<"$text" \
    || die "compose must run native Grok OIDC refresh"
  grep -Fq 'scripts.xai_oauth_refresh' <<<"$text" \
    || die "compose must run managed xAI OAuth refresh"
  if grep -Eq 'run_provider_status_observations_loop|DEFAULT_ENDPOINTS' <<<"$text"; then
    die "compose must not invoke the multi-provider observations loop"
  fi
}

preflight_image() {
  if ! "$docker_bin" image inspect "$image_name" >/dev/null 2>&1; then
    die "required image missing: ${image_name} (do not build from dirty source; load/promote the existing prod image)"
  fi
}

validate_native_credential_json() {
  local auth_file="$1"
  # Metadata-only validation; never print token values.
  python3 - "$auth_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001 - operator-facing preflight
    raise SystemExit(f"native credential is not valid JSON: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit("native credential payload must be a JSON object")

scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
record = payload.get(scope)
if not isinstance(record, dict):
    raise SystemExit(f"native credential missing expected Grok OIDC scope: {scope}")
if record.get("oidc_issuer") != "https://auth.x.ai":
    raise SystemExit("native credential has unexpected Grok OIDC issuer")
if record.get("oidc_client_id") != "b1a00492-073a-47ea-816f-4c329264a828":
    raise SystemExit("native credential has unexpected Grok OIDC client id")
if not record.get("expires_at"):
    raise SystemExit("native credential record missing expires_at")
if not record.get("refresh_token"):
    raise SystemExit("native credential record missing refresh token")
if not (record.get("key") or record.get("access_token")):
    raise SystemExit("native credential record missing current access credential")
print("native_credential_metadata_ok")
PY
}

validate_managed_credential_json() {
  # Metadata-only validation; never print token values.
  # The Python program is supplied with -c so stdin remains available for the
  # credential JSON from either a host file redirect or read-only docker pipe.
  python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception as exc:
    raise SystemExit(f"managed credential is not valid JSON: {exc}") from exc

if not isinstance(payload, dict):
    raise SystemExit("managed credential payload must be a JSON object")

scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
client = "b1a00492-073a-47ea-816f-4c329264a828"
record = payload.get(scope) if isinstance(payload.get(scope), dict) else None
if record is None and (
    payload.get("key") or payload.get("access_token") or payload.get("refresh_token")
):
    record = payload
if record is None:
    for value in payload.values():
        if isinstance(value, dict) and (
            value.get("key") or value.get("access_token") or value.get("refresh_token")
        ):
            record = value
            break
if not isinstance(record, dict):
    raise SystemExit("managed credential missing usable OAuth record")
cid = record.get("oidc_client_id") or record.get("client_id")
if cid != client:
    raise SystemExit("managed credential has unexpected OAuth client id")
# Managed may omit issuer; require access + refresh + expiry only.
if not record.get("expires_at"):
    raise SystemExit("managed credential record missing expires_at")
if not record.get("refresh_token"):
    raise SystemExit("managed credential record missing refresh token")
if not (record.get("key") or record.get("access_token")):
    raise SystemExit("managed credential record missing current access credential")
print("managed_credential_metadata_ok")
'
}

read_managed_credential_via_docker() {
  local auth_file="$1"
  local auth_dir auth_name
  auth_dir="$(dirname "$auth_file")"
  auth_name="$(basename "$auth_file")"
  # Read-only docker run with the existing prod image and host mount. Do not
  # mutate the credential or recreate proxies; exit non-zero on validation fail.
  "$docker_bin" run --rm --read-only \
    --security-opt no-new-privileges:true \
    --network none \
    --entrypoint python \
    -v "${auth_dir}:/credential:ro" \
    "$image_name" \
    -c "from pathlib import Path; import sys; p=Path('/credential/${auth_name}'); sys.stdout.write(p.read_text(encoding='utf-8'))"
}

preflight_native_credential() {
  local auth_file="$native_auth_file"
  [[ -f "$auth_file" ]] || die "native Grok OIDC credential missing: $auth_file"
  local mode_oct uid gid
  mode_oct="$(stat -c '%a' "$auth_file")"
  uid="$(stat -c '%u' "$auth_file")"
  gid="$(stat -c '%g' "$auth_file")"
  [[ "$mode_oct" == "600" ]] || die "native credential mode must be 0600, found ${mode_oct}: $auth_file"
  [[ "$uid" == "1000" && "$gid" == "1000" ]] \
    || die "native credential uid/gid must be 1000/1000, found ${uid}/${gid}: $auth_file"
  validate_native_credential_json "$auth_file"
}

preflight_managed_credential() {
  local auth_file="$managed_auth_file"
  [[ -e "$auth_file" ]] || die "managed xAI OAuth credential missing: $auth_file"
  [[ -f "$auth_file" ]] || die "managed xAI OAuth credential is not a regular file: $auth_file"
  local mode_oct uid gid
  mode_oct="$(stat -c '%a' "$auth_file")"
  uid="$(stat -c '%u' "$auth_file")"
  gid="$(stat -c '%g' "$auth_file")"
  [[ "$mode_oct" == "600" ]] || die "managed credential mode must be 0600, found ${mode_oct}: $auth_file"
  # Expected runtime ownership after sidecar write is 0/0. Existing nobody/
  # root-owned files are accepted for metadata preflight so status works before
  # the first apply cycle repairs ownership.
  if [[ "$uid" != "0" && "$uid" != "65534" ]]; then
    die "managed credential uid must be 0 (or legacy 65534), found ${uid}: $auth_file"
  fi
  if [[ "$gid" != "0" && "$gid" != "65534" ]]; then
    die "managed credential gid must be 0 (or legacy 65534), found ${gid}: $auth_file"
  fi

  if [[ -r "$auth_file" ]]; then
    validate_managed_credential_json <"$auth_file"
    return 0
  fi

  # Host user cannot read root/nobody-owned managed file: validate JSON safely
  # via a disposable read-only docker run against the existing prod image.
  info "managed credential unreadable by host user; validating via read-only docker mount"
  read_managed_credential_via_docker "$auth_file" \
    | validate_managed_credential_json \
    || die "failed to validate managed credential via docker preflight: $auth_file"
}

preflight_credentials() {
  preflight_native_credential
  preflight_managed_credential
}

print_proxy_snapshots() {
  local line container id started restarts
  while IFS= read -r line; do
    IFS='|' read -r container id started restarts <<<"$line"
    if [[ "$id" == "missing" || -z "$id" ]]; then
      info "proxy ${container}: missing"
    else
      info "proxy ${container}: id=${id} started_at=${started} restart_count=${restarts}"
    fi
  done < <(capture_proxy_snapshots)
}

print_sidecar_status() {
  local state health image started restarts
  if ! "$docker_bin" inspect "$container_name" >/dev/null 2>&1; then
    info "sidecar ${container_name}: not present"
    return 0
  fi
  state="$("$docker_bin" inspect -f '{{.State.Status}}' "$container_name")"
  health="$("$docker_bin" inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container_name")"
  image="$("$docker_bin" inspect -f '{{.Config.Image}}' "$container_name")"
  started="$("$docker_bin" inspect -f '{{.State.StartedAt}}' "$container_name")"
  restarts="$("$docker_bin" inspect -f '{{.RestartCount}}' "$container_name")"
  info "sidecar ${container_name}: state=${state} health=${health} image=${image} started_at=${started} restart_count=${restarts}"
}

run_status() {
  assert_compose_contract
  preflight_image
  preflight_credentials
  info "mode=status host_kind=$(detect_host_kind) compose=${compose_file} service=${service_name} image=${image_name}"
  info "native_auth_file=${native_auth_file}"
  info "managed_auth_file=${managed_auth_file}"
  print_sidecar_status
  print_proxy_snapshots
  info "status_ok"
}

run_apply() {
  require_wsl_host_for_apply
  local before_file after_file
  assert_compose_contract
  preflight_image
  preflight_credentials
  before_file="$(mktemp)"
  after_file="$(mktemp)"
  trap 'rm -f "$before_file" "$after_file"' RETURN
  capture_proxy_snapshots >"$before_file"
  assert_proxy_snapshots_present "$before_file"
  info "pre-apply proxy snapshots:"
  while IFS= read -r line; do info "  $line"; done <"$before_file"

  # Start/recreate only the dedicated dual-credential service. Never pass proxy
  # service names and never omit --no-deps/--no-build.
  compose up -d --no-deps --no-build "$service_name"
  wait_for_sidecar_healthy

  capture_proxy_snapshots >"$after_file"
  assert_proxy_snapshots_unchanged "$before_file" "$after_file"
  print_sidecar_status
  print_proxy_snapshots
  info "apply_ok service=${service_name} proxies_unchanged=true"
}

run_stop() {
  local before_file after_file
  assert_compose_contract
  before_file="$(mktemp)"
  after_file="$(mktemp)"
  trap 'rm -f "$before_file" "$after_file"' RETURN
  capture_proxy_snapshots >"$before_file"
  assert_proxy_snapshots_present "$before_file"
  info "pre-stop proxy snapshots:"
  while IFS= read -r line; do info "  $line"; done <"$before_file"

  # Stop only the dedicated service. --no-deps keeps unrelated units untouched.
  compose stop --timeout 30 "$service_name" || true
  # Prefer compose stop of the named service; if the container still exists in
  # a created state, leave it stopped without project-wide down.
  if "$docker_bin" inspect "$container_name" >/dev/null 2>&1; then
    local state
    state="$("$docker_bin" inspect -f '{{.State.Status}}' "$container_name")"
    if [[ "$state" == "running" ]]; then
      die "sidecar ${container_name} still running after compose stop"
    fi
  fi

  capture_proxy_snapshots >"$after_file"
  assert_proxy_snapshots_unchanged "$before_file" "$after_file"
  print_sidecar_status
  print_proxy_snapshots
  info "stop_ok service=${service_name} proxies_unchanged=true"
}

require_cmd "$docker_bin"
require_cmd python3
require_cmd stat
require_cmd mktemp
require_cmd grep

case "$mode" in
  status) run_status ;;
  apply) run_apply ;;
  stop) run_stop ;;
  *) die "unsupported mode: $mode" ;;
esac
