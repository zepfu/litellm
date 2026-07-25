"""Agent identity package-conversion contract tests.

The canonical implementation lives in
`litellm/integrations/aawm_agent_identity/`, with `__init__.py` preserving the
historical dotted import and monkeypatch surfaces while concern-specific
helpers live in sibling modules.

These tests pin the import, rebinding, script, and installed-wheel invariants
that the package layout must preserve.

See `.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md`
### Wave A1 for the full Impact Analysis / Test Spec / Source Spec.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

import pytest
from poetry.core.factory import Factory
from poetry.core.masonry.builders.wheel import WheelBuilder

REPO_ROOT = Path(__file__).resolve().parents[3]
WHEEL_BUILD_DIR = REPO_ROOT / ".wheel-build"
IDENTITY_PACKAGE_DIR = (
    REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity"
)

# The 12 names `scripts/backfill_rate_limit_observations.py` imports from the
# identity namespace (Wave A1 Test Spec: "13-symbol set" — the script actually
# imports these 12; kept as an explicit list so a future script edit that
# drops/adds a name fails this test loudly instead of silently).
_SCRIPT_IMPORT_SURFACE = (
    "_build_aawm_dsn",
    "_build_rate_limit_observations",
    "_ensure_session_history_schema",
    "_persist_session_history_records",
    "_rate_limit_storage_billing_period_end_at",
    "_rate_limit_storage_billing_period_start_at",
    "_rate_limit_storage_provider",
    "_rate_limit_storage_quota_limit",
    "_rate_limit_storage_quota_key",
    "_rate_limit_storage_quota_remaining",
    "_rate_limit_storage_quota_used",
    "_rate_limit_storage_remaining_pct",
)


def test_current_package_layout() -> None:
    """The converted package is canonical and the deleted module stays absent."""
    assert IDENTITY_PACKAGE_DIR.is_dir()
    assert not IDENTITY_PACKAGE_DIR.with_suffix(".py").exists()
    for module_name in (
        "__init__.py",
        "usage_extract.py",
        "provider_normalize.py",
        "request_signals.py",
        "prompt_overhead.py",
    ):
        assert (IDENTITY_PACKAGE_DIR / module_name).is_file()


def test_instance_import_path_stable() -> None:
    """Dotted import path + dedupe-match contract survive module->package conversion.

    Pins `pass_through_endpoints.py`'s canonical-instance dedupe check:
    `"aawm_agent_identity" in instance_module`.
    """
    from litellm.integrations.aawm_agent_identity import (
        AawmAgentIdentity,
        aawm_agent_identity_instance,
    )

    assert isinstance(aawm_agent_identity_instance, AawmAgentIdentity)
    instance_module = type(aawm_agent_identity_instance).__module__
    assert "aawm_agent_identity" in instance_module


def test_record_api_globals_host_is_identity_namespace() -> None:
    """`__globals__`-rebinding contract: record APIs live in the identity namespace.

    `_bind_session_history_record_apis()` rebinds record-API functions so
    their `__globals__` dict is THIS module's dict (not
    `aawm_session_history.record`'s), preserving monkeypatch-on-identity
    behavior for free-name helpers referenced inside those record APIs.
    """
    import litellm.integrations.aawm_agent_identity as identity_module

    fn = identity_module._build_failure_observation_only_record
    identity_dict = sys.modules["litellm.integrations.aawm_agent_identity"].__dict__
    assert fn.__globals__ is identity_dict

    # Monkeypatch a free-name helper on the identity namespace and confirm the
    # record API's behavior changes -- this is the live proof of the
    # __globals__-rebinding contract, not just an identity check.
    sentinel = {"called": False}
    original = identity_module._extract_agent_context

    def _stub_extract_agent_context(kwargs: Any) -> Any:
        sentinel["called"] = True
        return original(kwargs)

    identity_module._extract_agent_context = _stub_extract_agent_context
    try:
        assert fn.__globals__["_extract_agent_context"] is _stub_extract_agent_context
        # Call through the rebound global to prove the record API would
        # observe the patched helper if it invokes it by free name.
        fn.__globals__["_extract_agent_context"]({})
        assert sentinel["called"] is True
    finally:
        identity_module._extract_agent_context = original
        assert fn.__globals__["_extract_agent_context"] is original


def test_patch_surface_module_attrs_present() -> None:
    """Deliberate monkeypatch surfaces stay importable as module attributes."""
    import litellm.integrations.aawm_agent_identity as identity_module

    for name in ("threading", "queue", "time", "asyncio", "atexit", "importlib"):
        assert hasattr(identity_module, name), (
            f"expected deliberate patch-surface module attribute {name!r} on "
            "litellm.integrations.aawm_agent_identity"
        )


def test_scripts_import_surface() -> None:
    """`scripts/backfill_rate_limit_observations.py`'s import surface stays importable."""
    import litellm.integrations.aawm_agent_identity as identity_module

    backfill_script = REPO_ROOT / "scripts" / "backfill_rate_limit_observations.py"
    assert backfill_script.is_file(), f"missing: {backfill_script}"
    script_source = backfill_script.read_text(encoding="utf-8")

    missing_from_script = [name for name in _SCRIPT_IMPORT_SURFACE if name not in script_source]
    assert not missing_from_script, (
        "expected symbol(s) no longer referenced in "
        f"scripts/backfill_rate_limit_observations.py: {missing_from_script} "
        "-- update _SCRIPT_IMPORT_SURFACE in this test to match the script's "
        "actual import list"
    )

    missing_from_module = [name for name in _SCRIPT_IMPORT_SURFACE if not hasattr(identity_module, name)]
    assert not missing_from_module, (
        "litellm.integrations.aawm_agent_identity is missing symbol(s) that "
        f"scripts/backfill_rate_limit_observations.py imports: {missing_from_module}"
    )


def test_patched_repo_normalizer_intercepts_record_api() -> None:
    """Wave A2 `__globals__` contract, exercised NOW (must pass pre- and
    post-move) and again after the extraction: monkeypatching
    `_normalize_repository_identity` on the identity namespace intercepts a
    record API that consumes it via free-name lookup.

    `_build_session_history_record` -> `_extract_repository_identity_from_kwargs_with_source`
    -> `_extract_repository_identity_from_metadata_sources_with_source` ->
    `_normalize_repository_identity` is the live call chain today. After A2
    moves `_normalize_repository_identity` into `identity_repository.py`,
    the facade binding on the identity namespace must still be the object
    consulted by that chain (whether the chain itself also moves into
    `identity_repository.py` or stays reachable via the record-API
    `__globals__` rebind), so this test must keep passing unchanged.
    """
    import litellm.integrations.aawm_agent_identity as identity_module
    from litellm.integrations.aawm_agent_identity import _build_session_history_record

    sentinel = {"called_with": None}
    original = identity_module._normalize_repository_identity

    def _stub_normalize_repository_identity(value: Any) -> Any:
        sentinel["called_with"] = value
        return original(value)

    identity_module._normalize_repository_identity = _stub_normalize_repository_identity
    try:
        kwargs = {
            "litellm_call_id": "call-a2-repo-normalizer-probe",
            "model": "openai/gpt-5.4-mini",
            "custom_llm_provider": "openai",
            "call_type": "pass_through_endpoint",
            "litellm_params": {
                "metadata": {
                    "session_id": "session-a2-repo-normalizer-probe",
                    "repository": "litellm",
                    "trace_user_id": "codex",
                }
            },
            "messages": [{"role": "user", "content": "probe"}],
        }
        result = {
            "id": "resp-a2-repo-normalizer-probe",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "output": [],
        }

        record = _build_session_history_record(
            kwargs=kwargs,
            result=result,
            start_time=None,
            end_time=None,
        )

        assert record is not None
        assert sentinel["called_with"] is not None, (
            "expected _build_session_history_record's repository-extraction "
            "path to call the (patched) _normalize_repository_identity at "
            "least once via free-name / attribute lookup on the identity "
            "namespace"
        )
    finally:
        identity_module._normalize_repository_identity = original


@pytest.mark.integration
def test_installed_wheel_smoke(tmp_path: Path) -> None:
    """Build and import paired main/callback wheels outside the checkout."""
    if not WHEEL_BUILD_DIR.is_dir():
        pytest.skip(f".wheel-build directory not found at {WHEEL_BUILD_DIR}")

    smoke_root = tmp_path.resolve()
    assert not smoke_root.is_relative_to(REPO_ROOT.resolve())
    main_dist = smoke_root / "main-dist"
    callback_dist = smoke_root / "callback-dist"
    work_dir = smoke_root / "work"
    main_dist.mkdir()
    callback_dist.mkdir()
    work_dir.mkdir()

    main_wheel = WheelBuilder(Factory().create_poetry(REPO_ROOT)).build(
        target_dir=main_dist
    )
    assert main_wheel.is_file()
    build_env = dict(**__import__("os").environ)
    build_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(callback_dist),
            str(WHEEL_BUILD_DIR),
        ],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=300,
        env=build_env,
    )
    assert build_proc.returncode == 0, (
        "callback wheel build failed "
        f"(exit={build_proc.returncode}):\nstdout={build_proc.stdout}\n"
        f"stderr={build_proc.stderr}"
    )
    callback_wheels = sorted(
        callback_dist.glob("aawm_litellm_callbacks-*.whl")
    )
    assert len(callback_wheels) == 1, callback_wheels
    callback_wheel = callback_wheels[0]

    scratch_venv_dir = tmp_path / "aawm_wheel_smoke_venv"
    venv.EnvBuilder(
        with_pip=True,
        clear=True,
        system_site_packages=True,
    ).create(str(scratch_venv_dir))

    if sys.platform == "win32":  # pragma: no cover - not the target platform here
        venv_python = scratch_venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = scratch_venv_dir / "bin" / "python"
    assert venv_python.is_file(), f"scratch venv python missing: {venv_python}"

    install_proc = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-deps",
            str(main_wheel),
            str(callback_wheel),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(work_dir),
        env={key: value for key, value in build_env.items() if key != "PYTHONPATH"},
    )
    assert install_proc.returncode == 0, (
        "paired offline wheel install failed "
        f"(exit={install_proc.returncode}):\nstdout={install_proc.stdout}\n"
        f"stderr={install_proc.stderr}"
    )

    smoke_script = (
        "import importlib\n"
        "import importlib.metadata\n"
        "from pathlib import Path\n"
        "import litellm\n"
        "import aawm_litellm_callbacks.agent_identity as callback_identity\n"
        "import litellm.integrations.aawm_agent_identity as main_identity\n"
        f"repo_root = Path({str(REPO_ROOT.resolve())!r})\n"
        f"venv_root = Path({str(scratch_venv_dir.resolve())!r})\n"
        "module_names = (\n"
        "    'litellm.integrations.aawm_agent_identity.usage_extract',\n"
        "    'litellm.integrations.aawm_agent_identity.provider_normalize',\n"
        "    'litellm.integrations.aawm_agent_identity.request_signals',\n"
        "    'litellm.integrations.aawm_agent_identity.prompt_overhead',\n"
        ")\n"
        "modules = [importlib.import_module(name) for name in module_names]\n"
        "assert callback_identity.aawm_agent_identity_instance is "
        "main_identity.aawm_agent_identity_instance\n"
        "assert callback_identity._extract_agent_context is "
        "main_identity._extract_agent_context\n"
        "for module in [litellm, callback_identity, main_identity, *modules]:\n"
        "    module_path = Path(module.__file__).resolve()\n"
        "    assert not module_path.is_relative_to(repo_root), module_path\n"
        "    assert module_path.is_relative_to(venv_root), module_path\n"
        "for distribution_name in ('litellm', 'aawm-litellm-callbacks'):\n"
        "    distribution_path = Path(\n"
        "        importlib.metadata.distribution(distribution_name).locate_file('')\n"
        "    ).resolve()\n"
        "    assert distribution_path.is_relative_to(venv_root), distribution_path\n"
        "for module in modules:\n"
        "    assert callable(module.install)\n"
        "print('AAWM_PAIRED_WHEEL_SMOKE_OK')\n"
    )
    smoke_env = {
        key: value for key, value in build_env.items() if key != "PYTHONPATH"
    }
    run_proc = subprocess.run(
        [str(venv_python), "-c", smoke_script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(work_dir),
        env=smoke_env,
    )
    assert run_proc.returncode == 0, (
        "installed-wheel subprocess smoke failed "
        f"(exit={run_proc.returncode}):\nstdout={run_proc.stdout}\n"
        f"stderr={run_proc.stderr}"
    )
    assert "AAWM_PAIRED_WHEEL_SMOKE_OK" in run_proc.stdout
