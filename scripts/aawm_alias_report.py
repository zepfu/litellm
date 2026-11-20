"""Build a standalone AAWM alias routing report from validated config trees."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (  # noqa: E402
    compile_directory,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (  # noqa: E402
    RoutingSnapshot,
)
from scripts.aawm_alias_report_html import render_report  # noqa: E402

_DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class AliasSnapshot:
    """A compiled tree plus explicit runtime availability."""

    snapshot: RoutingSnapshot
    readiness_hash: str | None
    readiness_error: str | None


def _fetch_readiness_hash(url: str, timeout_seconds: float) -> str | None:
    """Return the active alias hash or None when readiness is unavailable."""
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    if response.status != 200:
        return None
    alias_status = payload.get("aawm_alias_config")
    if not isinstance(alias_status, dict):
        return None
    if alias_status.get("state") != "active":
        return None
    config_hash = alias_status.get("config_hash")
    return config_hash if isinstance(config_hash, str) else None


def _load_alias_snapshot(config_dir: Path, readiness_url: str, timeout_seconds: float) -> AliasSnapshot:
    snapshot = compile_directory(config_dir)
    readiness_hash = _fetch_readiness_hash(url=readiness_url, timeout_seconds=timeout_seconds)
    return AliasSnapshot(
        snapshot=snapshot,
        readiness_hash=readiness_hash,
        readiness_error=None if readiness_hash is not None else "unavailable",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the standalone HTML report.",
    )
    parser.add_argument(
        "--alpha-config-dir",
        type=Path,
        required=True,
        help="Alpha alias-config directory, including its active Compose overlays.",
    )
    parser.add_argument(
        "--production-config-dir",
        type=Path,
        required=True,
        help="Production alias-config directory copied from its deployed container.",
    )
    parser.add_argument(
        "--alpha-readiness-url",
        default="http://127.0.0.1:4011/health/readiness",
    )
    parser.add_argument(
        "--production-readiness-url",
        default="http://127.0.0.1:4000/health/readiness",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    alpha = _load_alias_snapshot(
        config_dir=args.alpha_config_dir,
        readiness_url=args.alpha_readiness_url,
        timeout_seconds=args.timeout_seconds,
    )
    production = _load_alias_snapshot(
        config_dir=args.production_config_dir,
        readiness_url=args.production_readiness_url,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(
            alpha=alpha.snapshot,
            production=production.snapshot,
            alpha_readiness_hash=alpha.readiness_hash,
            production_readiness_hash=production.readiness_hash,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
