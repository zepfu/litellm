"""RR-077: package portable Claude prompt paths into the harness bundle.

The release artifact must not embed machine-local absolute paths such as
`@/home/.../claude_acceptance_prompt.txt` inside config.json.

Coordinates with RR-079: source ``scripts/local-ci/config.json`` already uses
``@{config_dir}/...`` tokens. Packaging preserves those tokens, does not mutate
the on-disk source, and records manifest SHA-256 over the packaged bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "local-ci" / "build_harness_bundle.py"


def _load_module():
    name = "build_harness_bundle_rr077"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bhb():
    return _load_module()


def test_should_rewrite_absolute_local_at_paths_to_config_dir_placeholder(bhb) -> None:
    source_dir = bhb.LOCAL_CI_DIR
    absolute = (
        f"@{(source_dir / 'claude_acceptance_prompt.txt').resolve().as_posix()}"
    )
    rewritten = bhb._rewrite_at_path_token(absolute, source_dir)
    assert rewritten == "@{config_dir}/claude_acceptance_prompt.txt"
    assert "/home/" not in rewritten
    assert "zepfu" not in rewritten


def test_should_normalize_relative_at_paths_to_config_dir_placeholder(bhb) -> None:
    source_dir = Path("/tmp/fake-local-ci")
    assert (
        bhb._rewrite_at_path_token("@claude_acceptance_prompt.txt", source_dir)
        == "@{config_dir}/claude_acceptance_prompt.txt"
    )
    assert (
        bhb._rewrite_at_path_token("@./claude_acceptance_prompt_full_fanout.txt", source_dir)
        == "@{config_dir}/claude_acceptance_prompt_full_fanout.txt"
    )


def test_should_preserve_already_portable_and_foreign_absolute_paths(bhb) -> None:
    source_dir = bhb.LOCAL_CI_DIR
    portable = "@{config_dir}/claude_acceptance_prompt.txt"
    foreign = "@/tmp/elsewhere/not-local/prompt.txt"
    plain = "claude"
    assert bhb._rewrite_at_path_token(portable, source_dir) == portable
    assert bhb._rewrite_at_path_token(foreign, source_dir) == foreign
    assert bhb._rewrite_at_path_token(plain, source_dir) == plain


def test_should_rewrite_nested_command_lists_in_config_object(bhb, tmp_path: Path) -> None:
    source_dir = tmp_path / "local-ci"
    source_dir.mkdir()
    absolute_prompt = source_dir / "claude_acceptance_prompt.txt"
    absolute_full = source_dir / "claude_acceptance_prompt_full_fanout.txt"
    absolute_prompt.write_text("minimal\n", encoding="utf-8")
    absolute_full.write_text("full\n", encoding="utf-8")

    config = {
        "suite_version": 1,
        "claude": {
            "command": [
                "claude",
                "-p",
                f"@{absolute_prompt.resolve().as_posix()}",
                "--output-format",
                "json",
            ],
            "fanout_modes": {
                "minimal": {
                    "command": [
                        "claude",
                        "-p",
                        f"@{absolute_prompt.resolve().as_posix()}",
                    ]
                },
                "full": {
                    "command": [
                        "claude",
                        "-p",
                        f"@{absolute_full.resolve().as_posix()}",
                    ]
                },
            },
        },
        "untouched": {"path": "/tmp/keep-me", "note": "not an @-path"},
    }

    portable = bhb.rewrite_config_paths_for_bundle(config, source_config_dir=source_dir)
    assert (
        portable["claude"]["command"][2]
        == "@{config_dir}/claude_acceptance_prompt.txt"
    )
    assert (
        portable["claude"]["fanout_modes"]["minimal"]["command"][2]
        == "@{config_dir}/claude_acceptance_prompt.txt"
    )
    assert (
        portable["claude"]["fanout_modes"]["full"]["command"][2]
        == "@{config_dir}/claude_acceptance_prompt_full_fanout.txt"
    )
    assert portable["untouched"]["path"] == "/tmp/keep-me"
    # Source object must not be mutated in place.
    assert config["claude"]["command"][2].startswith("@/")


def test_should_render_repo_config_without_machine_home_paths(bhb) -> None:
    source_text = (bhb.LOCAL_CI_DIR / "config.json").read_text(encoding="utf-8")
    assert "/home/zepfu" not in source_text
    assert "@/home/" not in source_text

    payload = bhb.render_portable_config_bytes()
    text = payload.decode("utf-8")
    assert "/home/zepfu" not in text
    assert "projects/litellm/scripts/local-ci" not in text
    assert "@/home/" not in text
    cfg = json.loads(payload)
    assert cfg["claude"]["command"][2] == (
        "@{config_dir}/claude_acceptance_prompt.txt"
    )
    assert cfg["claude"]["fanout_modes"]["minimal"]["command"][2] == (
        "@{config_dir}/claude_acceptance_prompt.txt"
    )
    assert cfg["claude"]["fanout_modes"]["full"]["command"][2] == (
        "@{config_dir}/claude_acceptance_prompt_full_fanout.txt"
    )
    # Other harness keys remain intact.
    assert cfg.get("suite_version") is not None
    assert "codex" in cfg
    assert "gemini" in cfg


def test_should_package_portable_config_in_bundle_tarball(bhb, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    version = "rr077-test"
    artifact = bhb.build_bundle(version=version, outdir=outdir)
    assert artifact.is_file()
    assert artifact.name == f"litellm-local-ci-harness-{version}.tar.gz"

    prefix = f"litellm-local-ci-harness-{version}"
    with tarfile.open(artifact, "r:gz") as tar:
        names = tar.getnames()
        assert f"{prefix}/local-ci/config.json" in names
        assert f"{prefix}/MANIFEST.json" in names
        for relative_name in bhb.HARNESS_FILES:
            assert f"{prefix}/local-ci/{relative_name}" in names

        config_member = tar.extractfile(f"{prefix}/local-ci/config.json")
        assert config_member is not None
        config_bytes = config_member.read()
        config_text = config_bytes.decode("utf-8")
        assert "/home/zepfu" not in config_text
        cfg = json.loads(config_bytes)
        assert cfg["claude"]["command"][2] == (
            "@{config_dir}/claude_acceptance_prompt.txt"
        )
        assert cfg["claude"]["fanout_modes"]["full"]["command"][2] == (
            "@{config_dir}/claude_acceptance_prompt_full_fanout.txt"
        )

        # Prompt files remain present alongside the rewritten config.
        for prompt_name in (
            "claude_acceptance_prompt.txt",
            "claude_acceptance_prompt_full_fanout.txt",
        ):
            member = tar.extractfile(f"{prefix}/local-ci/{prompt_name}")
            assert member is not None
            assert member.read()

        manifest_member = tar.extractfile(f"{prefix}/MANIFEST.json")
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read())
        assert manifest["version"] == version
        files_by_path = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        assert files_by_path["local-ci/config.json"] == hashlib.sha256(
            config_bytes
        ).hexdigest()
        # With RR-079 portable source config, packaged bytes match source bytes.
        source_config_bytes = (bhb.LOCAL_CI_DIR / "config.json").read_bytes()
        assert config_bytes == source_config_bytes
        assert files_by_path["local-ci/config.json"] == hashlib.sha256(
            source_config_bytes
        ).hexdigest()
        # Manifest digests for non-config files still match source bytes.
        for relative_name in bhb.HARNESS_FILES:
            if relative_name == "config.json":
                continue
            source = (bhb.LOCAL_CI_DIR / relative_name).read_bytes()
            assert files_by_path[f"local-ci/{relative_name}"] == hashlib.sha256(
                source
            ).hexdigest()


def test_should_preserve_source_portable_config_without_mutation(bhb) -> None:
    """Combined RR-077/RR-079 contract: source is already portable; packaging is pure.

    Source config.json uses ``@{config_dir}/...`` tokens (RR-079). Bundle packaging
    must leave that on-disk file untouched and emit the same portable tokens in
    packaged bytes (no machine-home absolutization).
    """
    source_path = bhb.LOCAL_CI_DIR / "config.json"
    source = source_path.read_text(encoding="utf-8")
    assert "/home/zepfu" not in source
    assert "@/home/" not in source
    assert "@{config_dir}/claude_acceptance_prompt.txt" in source
    assert "@{config_dir}/claude_acceptance_prompt_full_fanout.txt" in source

    rendered = bhb.render_portable_config_bytes()
    after = source_path.read_text(encoding="utf-8")
    assert after == source  # pure function: no on-disk mutation
    assert rendered == source.encode("utf-8")  # already-portable source packages as-is

    cfg = json.loads(rendered)
    assert cfg["claude"]["command"][2] == "@{config_dir}/claude_acceptance_prompt.txt"
    assert (
        cfg["claude"]["fanout_modes"]["full"]["command"][2]
        == "@{config_dir}/claude_acceptance_prompt_full_fanout.txt"
    )
