#!/usr/bin/env bash
# Minimal, idempotent D1-622 transport manager for the litellm-dev loopback + VIP
# path. Root only.

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly SERVICE_IP="100.109.19.233"
readonly SERVICE_PORT="4001"
readonly SERVICE_CIDR="${SERVICE_IP}/32"
readonly TAILSCALE_SERVICE="svc:litellm-dev"
readonly TAILNET_SOURCE_CIDR="100.64.0.0/10"
readonly TAILNET_MARK="0x00040000"
readonly TAILNET_MARK_CLEAR_MASK="0xfffbffff"
readonly NMSLOT_CONN_NAME="litellm-dev-vip"
readonly NMSLOT_IFACE="litellm-dev-vip"
readonly NFT_TABLE="litellm_dev_transport"
readonly LISTEN_TIMEOUT_SECONDS="30"

print_error() {
  printf 'error: %s\n' "$*" >&2
}

die() {
  print_error "$*"
  exit 1
}

require_root() {
  [[ ${EUID:-$(id -u)} -eq 0 ]] || die "run as root: sudo $SCRIPT_NAME $*"
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "required command missing: ${cmd}"
}

usage() {
  cat <<'USAGE'
Usage:
  scripts/litellm-dev-transport.sh prepare
  scripts/litellm-dev-transport.sh activate
  scripts/litellm-dev-transport.sh status
  scripts/litellm-dev-transport.sh rollback
USAGE
}

nmcli_connection_value() {
  local profile="$1"
  local primary_field="$2"
  local fallback_field="${3:-}"

  local value
  value="$(nmcli -t -g "$primary_field" connection show "$profile" 2>/dev/null || true)"
  [[ "$value" == "--" ]] && value=""
  [[ -n "$value" ]] && { printf '%s\n' "$value"; return 0; }

  if [[ -n "$fallback_field" ]]; then
    value="$(nmcli -t -g "$fallback_field" connection show "$profile" 2>/dev/null || true)"
    [[ "$value" == "--" ]] && value=""
    [[ -n "$value" ]] && { printf '%s\n' "$value"; return 0; }
  fi

  return 1
}

nmcli_connection_exists() {
  local profile="$1"
  [[ -n "$(nmcli_connection_value "$profile" connection.id)" ]]
}

find_other_vip_owner() {
  local conn_id
  local conn_uuid
  local conn_addrs
  while IFS=':' read -r conn_id conn_uuid; do
    [[ "$conn_id" == "$NMSLOT_CONN_NAME" ]] && continue
    conn_addrs="$(nmcli_connection_value "$conn_uuid" ipv4.addresses)"
    if [[ -z "$conn_addrs" ]]; then
      continue
    fi
    if printf '%s\n' "$conn_addrs" | tr ',' '\n' | grep -Fxq "$SERVICE_CIDR"; then
      printf '%s\n' "$conn_id"
      return 0
    fi
  done < <(nmcli -t -f NAME,UUID connection show)
  return 1
}

validate_nic_ownership() {
  local owner
  owner="$(nmcli -g GENERAL.CONNECTION device show "$NMSLOT_IFACE" 2>/dev/null || true)"
  if [[ -n "$owner" && "$owner" != "$NMSLOT_CONN_NAME" ]]; then
    die "unexpected ${NMSLOT_IFACE} owner: ${owner}"
  fi
}

require_nmt_connection() {
  local existing
  existing="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.id || true)"
  if [[ -n "$existing" ]]; then
    local conn_type
    conn_type="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.type || true)"
    [[ -n "$conn_type" ]] || die "cannot resolve type for connection ${NMSLOT_CONN_NAME}"
    [[ "$conn_type" == "dummy" ]] || die "connection ${NMSLOT_CONN_NAME} is not dummy"

    local conn_iface
    conn_iface="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.interface-name || true)"
    if [[ -n "$conn_iface" && "$conn_iface" != "$NMSLOT_IFACE" ]]; then
      die "connection ${NMSLOT_CONN_NAME} bound to ${conn_iface}, expected ${NMSLOT_IFACE}"
    fi
  fi
}

apply_nic_config() {
  local existing_addr
  existing_addr="$(nmcli_connection_value "$NMSLOT_CONN_NAME" ipv4.addresses || true)"
  existing_addr="$(printf '%s\n' "$existing_addr" | tr ',' '\n' | sed '/^$/d')"

  nmcli connection add \
    type dummy \
    ifname "$NMSLOT_IFACE" \
    con-name "$NMSLOT_CONN_NAME" \
    ipv4.addresses "$SERVICE_CIDR" \
    ipv4.method manual \
    ipv4.never-default yes \
    ipv6.method ignore \
    autoconnect yes >/dev/null 2>&1 || true

  nmcli connection modify "$NMSLOT_CONN_NAME" \
    connection.interface-name "$NMSLOT_IFACE" \
    ipv4.addresses "$SERVICE_CIDR" \
    ipv4.method manual \
    ipv4.never-default yes \
    ipv6.method ignore \
    autoconnect yes >/dev/null

  if ! grep -Fxq "$SERVICE_CIDR" <<<"$existing_addr"; then
    printf 'prepared connection %s with service VIP %s\n' "$NMSLOT_CONN_NAME" "$SERVICE_CIDR"
  fi

  nmcli connection up "$NMSLOT_CONN_NAME" >/dev/null
}

assert_vip_present() {
  if ! ip -o -4 addr show "$NMSLOT_IFACE" >/dev/null 2>&1; then
    die "dummy interface ${NMSLOT_IFACE} is missing"
  fi

  ip -o -4 addr show "$NMSLOT_IFACE" | awk '{print $4}' | grep -Fxq "$SERVICE_CIDR" \
    || die "dummy interface ${NMSLOT_IFACE} does not expose ${SERVICE_CIDR}"
}

render_nft_rules() {
  local file="$1"

  cat >"$file" <<EOF
table inet ${NFT_TABLE} {
  chain pre_dnat {
    type filter hook prerouting priority -300; policy accept;
    ip daddr ${SERVICE_IP} tcp dport ${SERVICE_PORT} iifname != "tailscale0" drop
  }

  chain postroute_mark_clear {
    type filter hook postrouting priority mangle; policy accept;
    ip saddr ${TAILNET_SOURCE_CIDR} meta l4proto tcp \
      ct original ip daddr ${SERVICE_IP} ct original proto-dst ${SERVICE_PORT} \
      meta mark & ${TAILNET_MARK} == ${TAILNET_MARK} \
      meta mark set meta mark & ${TAILNET_MARK_CLEAR_MASK}
  }
}
EOF
}

apply_nft_rules() {
  local rules_file
  rules_file="$(mktemp)"
  render_nft_rules "$rules_file"

  if ! nft --check -f "$rules_file"; then
    rm -f "$rules_file"
    die "generated nft rules failed nft --check"
  fi

  if nft list table inet "$NFT_TABLE" >/dev/null 2>&1; then
    nft delete table inet "$NFT_TABLE" >/dev/null
  fi

  nft -f "$rules_file" >/dev/null
  rm -f "$rules_file"
}

socket_listen_check() {
  local host="$1" port="$2"
  local deadline=$((SECONDS + LISTEN_TIMEOUT_SECONDS))

  while ((SECONDS < deadline)); do
    if python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 1
  done

  die "listener not ready on ${host}:${port}"
}

rollback_tailscale() {
  tailscale serve --service="${TAILSCALE_SERVICE}" --tcp "${SERVICE_PORT}" "tcp://127.0.0.1:${SERVICE_PORT}" --yes
}

status_listener() {
  printf 'listener: '
  if python3 - "$SERVICE_IP" "$SERVICE_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
else:
    print('ok')
finally:
    sock.close()
PY
  then
    printf 'ready\n'
  else
    printf 'not ready\n'
  fi
}

status_nic() {
  local owner
  owner="$(nmcli -g GENERAL.CONNECTION device show "$NMSLOT_IFACE" 2>/dev/null || true)"

  local conn_id conn_uuid conn_type conn_iface conn_addrs
  conn_id="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.id || true)"
  conn_uuid="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.uuid || true)"
  conn_type="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.type || true)"
  conn_iface="$(nmcli_connection_value "$NMSLOT_CONN_NAME" connection.interface-name || true)"
  conn_addrs="$(nmcli_connection_value "$NMSLOT_CONN_NAME" ipv4.addresses || true)"

  if [[ -n "$conn_id" ]]; then
    printf 'nm-connection: name=%s uuid=%s type=%s iface=%s addresses=%s\n' \
      "$conn_id" "${conn_uuid:-<none>}" "${conn_type:-<none>}" "${conn_iface:-<none>}" "${conn_addrs:-<none>}"
    printf 'nm-device %s -> %s\n' "$NMSLOT_IFACE" "${owner:-<none>}"
  else
    printf 'nm-connection: missing\n'
    printf 'nm-device %s -> %s\n' "$NMSLOT_IFACE" "${owner:-<none>}"
  fi
}

status_nft() {
  if nft list table inet "$NFT_TABLE" >/dev/null 2>&1; then
    printf 'nft-table: %s present\n' "$NFT_TABLE"
  else
    printf 'nft-table: unavailable or absent\n'
  fi
}

status_tailscale() {
  if tailscale serve status >/dev/null 2>&1; then
    printf 'tailscale-serve: available\n'
    tailscale serve status 2>/dev/null | sed -n '1,80p'
  else
    printf 'tailscale-serve: unavailable (daemon missing/permissioned)\n'
  fi
}

cmd_prepare() {
  require_root
  require_cmd nmcli
  require_cmd nft
  require_cmd ip

  local vip_owner
  vip_owner="$(find_other_vip_owner || true)"
  if [[ -n "$vip_owner" ]]; then
    die "service VIP ${SERVICE_CIDR} already owned by connection ${vip_owner}"
  fi

  validate_nic_ownership
  require_nmt_connection
  apply_nic_config
  assert_vip_present
  apply_nft_rules

  printf 'prepare complete: %s %s\n' "$NMSLOT_CONN_NAME" "$NMSLOT_IFACE"
}

cmd_activate() {
  require_root
  require_cmd tailscale

  socket_listen_check "$SERVICE_IP" "$SERVICE_PORT"
  tailscale serve --service="${TAILSCALE_SERVICE}" --tun --tcp "${SERVICE_PORT}" --yes
  printf 'activate complete: %s\n' "$TAILSCALE_SERVICE"
}

cmd_status() {
  status_listener
  status_nic
  status_nft
  status_tailscale
}

cmd_rollback() {
  require_root
  require_cmd tailscale

  rollback_tailscale
  printf 'rollback complete: %s is mapped to tcp://127.0.0.1:%s\n' "$TAILSCALE_SERVICE" "$SERVICE_PORT"
  printf 'rollback retained dummy VIP %s and nft guard for future litellm-dev recreates\n' "$SERVICE_CIDR"
}

main() {
  case "${1:-}" in
    prepare)
      cmd_prepare
      ;;
    activate)
      cmd_activate
      ;;
    status)
      cmd_status
      ;;
    rollback)
      cmd_rollback
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
