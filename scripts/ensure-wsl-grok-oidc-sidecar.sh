#!/usr/bin/env bash
# Ensure the WSL-local single-writer Grok OIDC refresh sidecar.
#
# Default: --status
# Mutating modes start/stop only the dedicated service from
# docker-compose.wsl-grok-oidc.yml and refuse to touch aawm-litellm or
# litellm-dev. Proxy identity (container ID, start timestamp, restart count)
# is snapshotted before and after every mutation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${WSL_GROK_OIDC_COMPOSE_FILE:-${repo_root}/docker-compose.wsl-grok-oidc.yml}"
service_name="${WSL_GROK_OIDC_SERVICE_NAME:-wsl-grok-oidc-refresh}"
container_name="${WSL_GROK_OIDC_CONTAINER_NAME:-aawm-wsl-grok-oidc-refresh}"
image_name="${WSL_GROK_OIDC_IMAGE:-aawm-provider-status-observations:prod}"
auth_file="${WSL_GROK_OIDC_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
docker_bin="${WSL_GROK_OIDC_DOCKER_BIN:-docker}"

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

  --status  Preflight image + credential, report sidecar + proxy snapshots
            (default).
  --apply   Start/recreate only the dedicated WSL Grok OIDC service with
            --no-deps --no-build after proving both LiteLLM proxies stay
            unchanged.
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
    || die "compose must mount only /home/zepfu/.grok"
  # Refuse accidental multi-service or proxy coupling in this file.
  if grep -Eq 'aawm-litellm|litellm-dev' <<<"$text"; then
    die "compose file must not reference aawm-litellm or litellm-dev"
  fi
  if grep -Eq '/home/zepfu/\\.codex|/home/zepfu/\\.litellm/xai|/home/zepfu/\\.kimi-code|/home/zepfu/\\.alibaba' <<<"$text"; then
    die "compose must not mount Codex, managed xAI, Kimi, or Alibaba credential dirs"
  fi
  grep -Fq 'AAWM_GROK_OIDC_REFRESH_ENABLED=1' <<<"$text" \
    || die "compose must enable Grok OIDC refresh"
  grep -Fq 'AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=300' <<<"$text" \
    || die "compose must set Grok refresh interval 300"
  grep -Fq 'AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS=900' <<<"$text" \
    || die "compose must set Grok refresh buffer 900"
  grep -Fq 'AAWM_GROK_OIDC_FORCE_REFRESH=0' <<<"$text" \
    || die "compose must set Grok force refresh 0"
  grep -Fq 'AAWM_PROVIDER_STATUS_APPLY=0' <<<"$text" \
    || die "compose must disable provider-status DB apply"
  grep -Fq 'AAWM_CODEX_OAUTH_REFRESH_ENABLED=0' <<<"$text" \
    || die "compose must disable Codex OAuth refresh"
  grep -Fq 'AAWM_XAI_OAUTH_REFRESH_ENABLED=0' <<<"$text" \
    || die "compose must disable managed xAI OAuth refresh"
  grep -Fq 'AAWM_KIMI_OAUTH_REFRESH_ENABLED=0' <<<"$text" \
    || die "compose must disable Kimi OAuth refresh"
  grep -Fq 'AAWM_GROK_BILLING_POLL_ENABLED=0' <<<"$text" \
    || die "compose must disable Grok billing poll"
  grep -Fq 'AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED=0' <<<"$text" \
    || die "compose must disable anomaly scan"
  grep -Fq 'scripts.grok_oidc_refresh' <<<"$text" \
    || die "compose must run a Grok-only refresh loop"
  if grep -Eq 'run_provider_status_observations_loop|DEFAULT_ENDPOINTS' <<<"$text"; then
    die "compose must not invoke the multi-provider observations loop"
  fi
}

preflight_image() {
  if ! "$docker_bin" image inspect "$image_name" >/dev/null 2>&1; then
    die "required image missing: ${image_name} (do not build from dirty source; load/promote the existing prod image)"
  fi
}

preflight_credential() {
  [[ -f "$auth_file" ]] || die "Grok OIDC credential missing: $auth_file"
  local mode_oct uid gid
  mode_oct="$(stat -c '%a' "$auth_file")"
  uid="$(stat -c '%u' "$auth_file")"
  gid="$(stat -c '%g' "$auth_file")"
  [[ "$mode_oct" == "600" ]] || die "credential mode must be 0600, found ${mode_oct}: $auth_file"
  [[ "$uid" == "1000" && "$gid" == "1000" ]] \
    || die "credential uid/gid must be 1000/1000, found ${uid}/${gid}: $auth_file"
  # Metadata-only validation; never print token values.
  python3 - "$auth_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001 - operator-facing preflight
    raise SystemExit(f"credential is not valid JSON: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit("credential payload must be a JSON object")

scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
record = payload.get(scope)
if not isinstance(record, dict):
    raise SystemExit(f"credential missing expected Grok OIDC scope: {scope}")
if record.get("oidc_issuer") != "https://auth.x.ai":
    raise SystemExit("credential has unexpected Grok OIDC issuer")
if record.get("oidc_client_id") != "b1a00492-073a-47ea-816f-4c329264a828":
    raise SystemExit("credential has unexpected Grok OIDC client id")
if not record.get("expires_at"):
    raise SystemExit("credential record missing expires_at")
if not record.get("refresh_token"):
    raise SystemExit("credential record missing refresh token")
if not record.get("key"):
    raise SystemExit("credential record missing current access credential")
print("credential_metadata_ok")
PY
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
  preflight_credential
  info "mode=status compose=${compose_file} service=${service_name} image=${image_name}"
  print_sidecar_status
  print_proxy_snapshots
  info "status_ok"
}

run_apply() {
  local before_file after_file
  assert_compose_contract
  preflight_image
  preflight_credential
  before_file="$(mktemp)"
  after_file="$(mktemp)"
  trap 'rm -f "$before_file" "$after_file"' RETURN
  capture_proxy_snapshots >"$before_file"
  assert_proxy_snapshots_present "$before_file"
  info "pre-apply proxy snapshots:"
  while IFS= read -r line; do info "  $line"; done <"$before_file"

  # Start/recreate only the dedicated Grok OIDC service. Never pass proxy
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
