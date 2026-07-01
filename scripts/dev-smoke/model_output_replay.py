#!/usr/bin/env python3
"""Local dev smoke: replay sanitized malformed model output through production paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))

from support.model_output_replay import (  # noqa: E402
    ALL_LANES,
    ManifestValidationError,
    ReplayLane,
    compact_public_result,
    default_fixtures_dir,
    resolve_fixture,
    run_replay,
)


def _parse_lanes(raw: str | None) -> list[ReplayLane]:
    if not raw or raw.strip().lower() == "all":
        return list(ALL_LANES)
    lanes: list[ReplayLane] = []
    for part in raw.split(","):
        lane = part.strip().lower()
        if lane not in ALL_LANES:
            raise argparse.ArgumentTypeError(f"unknown lane: {part!r}")
        if lane not in lanes:
            lanes.append(lane)  # type: ignore[arg-type]
    return lanes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_fixtures_dir() / "manifest.jsonl",
        help="Path to manifest.jsonl (default: tests/fixtures/model_output_replay/manifest.jsonl)",
    )
    parser.add_argument(
        "--fixture",
        required=True,
        help="fixture_id from manifest",
    )
    parser.add_argument(
        "--lane",
        default="all",
        help="Comma-separated lanes: detect,repair,intake,scorer or all",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit compact JSON only (no raw transcript fields)",
    )
    args = parser.parse_args(argv)

    try:
        lanes = _parse_lanes(args.lane)
        loaded = resolve_fixture(
            manifest_path=args.manifest,
            fixture_id=args.fixture,
            fixtures_dir=args.manifest.parent,
        )
        result = compact_public_result(run_replay(loaded, lanes))
    except (ManifestValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True))  # noqa: T201
    else:
        print(f"fixture_id={result.get('fixture_id')}")  # noqa: T201
        print(f"disposition={result.get('disposition')}")  # noqa: T201
        print(f"expected_disposition={result.get('expected_disposition')}")  # noqa: T201
        for lane_name, lane_result in (result.get("lanes") or {}).items():
            print(f"lane={lane_name} " + json.dumps(lane_result, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
