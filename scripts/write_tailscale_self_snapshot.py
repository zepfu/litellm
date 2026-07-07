#!/usr/bin/env python3
"""Write a sanitized Tailscale self identity snapshot for local host attribution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _tailscale_cgnat_ipv4(values: Any) -> List[str]:
    import ipaddress

    network = ipaddress.ip_network("100.64.0.0/10")
    ips: List[str] = []
    if not isinstance(values, (list, tuple)):
        return ips
    for raw in values:
        try:
            parsed = ipaddress.ip_address(str(raw).strip())
        except ValueError:
            continue
        if parsed.version == 4 and parsed in network:
            text = str(parsed)
            if text not in ips:
                ips.append(text)
    return ips


def build_sanitized_tailscale_self_snapshot(
    status_payload: Dict[str, Any],
) -> Dict[str, Any]:
    self_section = status_payload.get("Self")
    if not isinstance(self_section, dict):
        self_section = {}

    magic_dns_suffix = _clean_str(status_payload.get("MagicDNSSuffix"))
    current_tailnet = status_payload.get("CurrentTailnet")
    if not magic_dns_suffix and isinstance(current_tailnet, dict):
        magic_dns_suffix = _clean_str(current_tailnet.get("MagicDNSSuffix"))

    dns_name = _clean_str(self_section.get("DNSName"))
    if dns_name:
        dns_name = dns_name.rstrip(".")

    tailscale_ips = _tailscale_cgnat_ipv4(self_section.get("TailscaleIPs"))
    if not tailscale_ips:
        tailscale_ips = _tailscale_cgnat_ipv4(status_payload.get("TailscaleIPs"))

    snapshot: Dict[str, Any] = {
        "schema_version": 1,
        "self_dns_name": dns_name,
        "self_tailscale_ips": tailscale_ips,
        "magic_dns_suffix": magic_dns_suffix,
    }
    return snapshot


def fetch_tailscale_status_json() -> Dict[str, Any]:
    completed = subprocess.run(
        ["tailscale", "status", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/home/zepfu/.analysis/aawm-host-identity/tailscale-self.json",
        help="Path for the sanitized snapshot JSON file.",
    )
    parser.add_argument(
        "--input",
        help="Optional existing tailscale status JSON file instead of running tailscale.",
    )
    args = parser.parse_args(argv)

    if args.input:
        status_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        try:
            status_payload = fetch_tailscale_status_json()
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"failed to load tailscale status: {exc}\n")
            return 1

    snapshot = build_sanitized_tailscale_self_snapshot(status_payload)
    if not snapshot.get("self_dns_name") and not snapshot.get("self_tailscale_ips"):
        sys.stderr.write("tailscale status missing Self.DNSName and TailscaleIPs\n")
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.replace(temp_name, output_path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    sys.stdout.write(f"{output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
