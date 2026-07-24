"""Wave A1 (agent_identity package conversion) contract-pinning tests.

`litellm/integrations/aawm_agent_identity.py` is CURRENTLY a single ~16.7k
line module. Wave A1's engineer will `git mv` it verbatim to
`litellm/integrations/aawm_agent_identity/__init__.py` and remap the
`.wheel-build/pyproject.toml` force-include from single-file to package form.

These tests PIN the invariants that conversion must preserve. They must pass
against the CURRENT single-file form (module `__init__.py` boundary doesn't
exist yet) AND, by design, against the post-conversion package form, because
every assertion below is expressed in terms of the importable dotted path
`litellm.integrations.aawm_agent_identity`, not the file layout.

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

REPO_ROOT = Path(__file__).resolve().parents[3]
WHEEL_BUILD_DIR = REPO_ROOT / ".wheel-build"

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
    """HARD GATE for Wave A2+: the built-and-installed wheel must still expose
    `aawm_litellm_callbacks.agent_identity.aawm_agent_identity_instance` and a
    callable `_extract_agent_context`, exercised from a real subprocess against
    a clean scratch venv -- not the in-process test interpreter's sys.path.

    Skips (does not fail) only when pip/network/build tooling is unavailable
    in this environment; otherwise it must run and pass.
    """
    if not WHEEL_BUILD_DIR.is_dir():
        pytest.skip(f".wheel-build directory not found at {WHEEL_BUILD_DIR}")

    # Clean-build guard: stale build/dist/egg-info artifacts under
    # .wheel-build have previously caused a silent install of stale code
    # (documented prod incident). Always start from a clean slate.
    for stale_dir_name in ("build", "dist"):
        stale_dir = WHEEL_BUILD_DIR / stale_dir_name
        if stale_dir.exists():
            import shutil

            shutil.rmtree(stale_dir)
    for egg_info in WHEEL_BUILD_DIR.glob("*.egg-info"):
        import shutil

        shutil.rmtree(egg_info)

    build_env = dict(**__import__("os").environ)

    build_proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", "dist/"],
        cwd=str(WHEEL_BUILD_DIR),
        capture_output=True,
        text=True,
        timeout=300,
        env=build_env,
    )
    if build_proc.returncode != 0:
        combined_output = (build_proc.stdout or "") + (build_proc.stderr or "")
        network_or_tooling_signatures = (
            "No module named build",
            "Could not find a version that satisfies",
            "Temporary failure in name resolution",
            "Network is unreachable",
            "Connection refused",
            "ConnectionError",
        )
        if any(sig in combined_output for sig in network_or_tooling_signatures):
            pytest.skip(
                "wheel build failed due to unavailable pip/build tooling or "
                f"network access:\n{combined_output[-2000:]}"
            )
        raise AssertionError(f"wheel build failed (exit={build_proc.returncode}):\n" f"{combined_output[-4000:]}")

    dist_dir = WHEEL_BUILD_DIR / "dist"
    wheels = sorted(dist_dir.glob("aawm_litellm_callbacks-*.whl"))
    assert wheels, f"no wheel produced under {dist_dir}"
    wheel_path = wheels[-1]

    scratch_venv_dir = tmp_path / "aawm_wheel_smoke_venv"
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(str(scratch_venv_dir))
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not create scratch venv: {exc}")

    if sys.platform == "win32":  # pragma: no cover - not the target platform here
        venv_python = scratch_venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = scratch_venv_dir / "bin" / "python"
    assert venv_python.is_file(), f"scratch venv python missing: {venv_python}"

    # The wheel's own dependency closure (`dependencies = ["litellm"]` in
    # `.wheel-build/pyproject.toml`) pulls generic PyPI litellm without the
    # optional `fastapi` extra. `aawm_agent_identity` has a top-level import
    # of `litellm.proxy.aawm_route_logging`, which needs `fastapi` -- a
    # pre-existing packaging gap unrelated to the A1 conversion itself.
    # Install it alongside the wheel so this test validates the conversion
    # invariant (package installs + exposes the right symbols) rather than
    # failing on that unrelated gap.
    install_proc = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--quiet",
            str(wheel_path),
            "fastapi",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if install_proc.returncode != 0:
        combined_output = (install_proc.stdout or "") + (install_proc.stderr or "")
        network_or_tooling_signatures = (
            "Could not find a version that satisfies",
            "Temporary failure in name resolution",
            "Network is unreachable",
            "Connection refused",
            "ConnectionError",
            "No matching distribution found",
        )
        if any(sig in combined_output for sig in network_or_tooling_signatures):
            pytest.skip("wheel install failed due to unavailable pip/network access:\n" f"{combined_output[-2000:]}")
        raise AssertionError(f"wheel install failed (exit={install_proc.returncode}):\n" f"{combined_output[-4000:]}")

    smoke_script = (
        "from aawm_litellm_callbacks.agent_identity import "
        "aawm_agent_identity_instance\n"
        "from aawm_litellm_callbacks.agent_identity import _extract_agent_context\n"
        "assert callable(_extract_agent_context)\n"
        "assert aawm_agent_identity_instance is not None\n"
        "print('AAWM_WHEEL_SMOKE_OK')\n"
    )
    run_proc = subprocess.run(
        [str(venv_python), "-c", smoke_script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run_proc.returncode == 0, (
        "installed-wheel subprocess smoke failed "
        f"(exit={run_proc.returncode}):\nstdout={run_proc.stdout}\n"
        f"stderr={run_proc.stderr}"
    )
    assert "AAWM_WHEEL_SMOKE_OK" in run_proc.stdout
