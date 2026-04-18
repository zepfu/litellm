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


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "model-config-version.txt"
SOURCE_FILE = ROOT / "model_prices_and_context_window.json"
DEFAULT_OUTDIR = ROOT / "dist"


def _read_version() -> str:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"Empty model config version file: {VERSION_FILE}")
    return version


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _add_bytes_to_tar(tar: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = int(datetime.now(tz=timezone.utc).timestamp())
    tar.addfile(info, io.BytesIO(payload))


def build_bundle(version: str, outdir: Path) -> Path:
    source_bytes = SOURCE_FILE.read_bytes()
    source_data = json.loads(source_bytes)
    artifact_prefix = f"litellm-model-config-{version}"
    artifact_path = outdir / f"{artifact_prefix}.tar.gz"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "artifact": artifact_path.name,
        "version": version,
        "built_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_file": SOURCE_FILE.name,
        "source_sha256": _sha256_bytes(source_bytes),
        "model_count": len(source_data) if isinstance(source_data, dict) else None,
    }

    with tarfile.open(artifact_path, "w:gz") as tar:
        tar.add(SOURCE_FILE, arcname=f"{artifact_prefix}/{SOURCE_FILE.name}")
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        _add_bytes_to_tar(tar, f"{artifact_prefix}/MANIFEST.json", manifest_bytes)

    return artifact_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the standalone LiteLLM model config archive."
    )
    parser.add_argument("--version", default=None)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    version = args.version or _read_version()
    artifact_path = build_bundle(version=version, outdir=Path(args.outdir))
    print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
