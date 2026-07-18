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
CONFIG_FILE_NAME = "config.json"
CONFIG_DIR_PLACEHOLDER = "{config_dir}"


def _read_version() -> str:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"Empty harness version file: {VERSION_FILE}")
    return version


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _add_bytes_to_tar(
    tar: tarfile.TarFile,
    arcname: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = int(datetime.now(tz=timezone.utc).timestamp())
    tar.addfile(info, io.BytesIO(payload))


def _rewrite_at_path_token(token: str, source_config_dir: Path) -> str:
    """Rewrite a single @-prefixed path token for a portable harness bundle.

    Absolute paths under *source_config_dir* become ``@{config_dir}/<rel>`` so
    the shipped artifact does not embed a machine-local home path. Relative
    ``@`` tokens are normalized onto the same placeholder form. Non-local
    absolute ``@`` paths are left unchanged.
    """
    if not token.startswith("@"):
        return token

    path_part = token[1:]
    if not path_part or path_part.startswith(f"{CONFIG_DIR_PLACEHOLDER}/"):
        return token

    candidate = Path(path_part)
    source_config_dir = source_config_dir.resolve()

    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(source_config_dir)
        except ValueError:
            return token
        return f"@{CONFIG_DIR_PLACEHOLDER}/{relative.as_posix()}"

    # Relative @path (or @./path) → @{config_dir}/path
    relative = Path(path_part)
    # Drop a leading "./" for stable portable form.
    relative_posix = relative.as_posix()
    if relative_posix.startswith("./"):
        relative_posix = relative_posix[2:]
    return f"@{CONFIG_DIR_PLACEHOLDER}/{relative_posix}"


def rewrite_config_paths_for_bundle(
    config: Any,
    source_config_dir: Path | None = None,
) -> Any:
    """Deep-copy config data, rewriting Claude prompt @-paths for standalone use."""
    config_dir = (source_config_dir or LOCAL_CI_DIR).resolve()

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, str):
            return _rewrite_at_path_token(value, config_dir)
        return value

    return walk(config)


def render_portable_config_bytes(
    source_config_path: Path | None = None,
    source_config_dir: Path | None = None,
) -> bytes:
    """Load config.json and return portable JSON bytes for the release bundle."""
    config_path = source_config_path or (LOCAL_CI_DIR / CONFIG_FILE_NAME)
    config_dir = source_config_dir or config_path.parent
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    portable = rewrite_config_paths_for_bundle(raw, source_config_dir=config_dir)
    return (json.dumps(portable, indent=2, sort_keys=False) + "\n").encode("utf-8")


def build_bundle(version: str, outdir: Path) -> Path:
    config_path = LOCAL_CI_DIR / CONFIG_FILE_NAME
    config_bytes = render_portable_config_bytes(
        source_config_path=config_path,
        source_config_dir=LOCAL_CI_DIR,
    )
    suite_version = json.loads(config_bytes.decode("utf-8")).get("suite_version")
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
            if relative_name == CONFIG_FILE_NAME:
                payload = config_bytes
            else:
                payload = source_path.read_bytes()
            digest = _sha256_bytes(payload)
            _add_bytes_to_tar(tar, arcname, payload)

            manifest["files"].append(
                {
                    "path": f"local-ci/{relative_name}",
                    "sha256": digest,
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
