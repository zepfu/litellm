#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCAL_CI_DIR = ROOT / "scripts" / "local-ci"
VERSION_FILE = LOCAL_CI_DIR / "harness-version.txt"
DEFAULT_OUTDIR = ROOT / "dist"
HARNESS_FILES = [
    "README.md",
    "run_acceptance.sh",
    "run_acceptance.py",
    "compare_artifacts.py",
    "config.json",
    "claude_acceptance_prompt.txt",
    "claude_acceptance_prompt_full_fanout.txt",
]


def _read_version() -> str:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"Empty harness version file: {VERSION_FILE}")
    return version


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_bytes_to_tar(
    tar: tarfile.TarFile,
    arcname: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = int(datetime.now(tz=timezone.utc).timestamp())
    tar.addfile(info, io.BytesIO(payload))


def build_bundle(version: str, outdir: Path) -> Path:
    config = json.loads((LOCAL_CI_DIR / "config.json").read_text(encoding="utf-8"))
    suite_version = config.get("suite_version")
    prefix = f"litellm-local-ci-harness-{version}"
    artifact_path = outdir / f"{prefix}.tar.gz"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "artifact": artifact_path.name,
        "version": version,
        "suite_version": suite_version,
        "built_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": [],
    }

    with tarfile.open(artifact_path, "w:gz") as tar:
        for relative_name in HARNESS_FILES:
            source_path = LOCAL_CI_DIR / relative_name
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing harness file: {source_path}")

            arcname = f"{prefix}/local-ci/{relative_name}"
            tar.add(source_path, arcname=arcname)
            manifest["files"].append(
                {
                    "path": f"local-ci/{relative_name}",
                    "sha256": _sha256_file(source_path),
                }
            )

        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _add_bytes_to_tar(tar, f"{prefix}/MANIFEST.json", manifest_bytes)

    return artifact_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the standalone LiteLLM local acceptance harness bundle."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Override the harness version. Defaults to scripts/local-ci/harness-version.txt.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Output directory for the compressed harness artifact.",
    )
    args = parser.parse_args()

    version = args.version or _read_version()
    artifact_path = build_bundle(version=version, outdir=Path(args.outdir))
    print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
