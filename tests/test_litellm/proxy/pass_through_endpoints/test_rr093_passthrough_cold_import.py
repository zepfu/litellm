"""RR-093 / RR-118: passthrough adapter-package import cycle and duplicate import.

RR-093: ``aawm_adapter_runtime/__init__.py`` eagerly imports
``anthropic_adapter_calls``, which module-load imports
``PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES`` from
``pass_through_endpoints``. That cycle breaks a fresh interpreter import of
the passthrough modules and prevents routing tests from collecting.

Break the cycle inside the adapter package (stop the eager package import
*or* stop the module-load parent import). Do not rewrite passthrough
initialization. Prefer not editing ``pass_through_endpoints.py``.

RR-118: ``response_utils.py`` must keep the module-scope watermark import
and must not repeat it inside ``_build_responses_response_from_adapter_response``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ADAPTER_PACKAGE_DIR = (
    REPO_ROOT / "litellm/proxy/pass_through_endpoints/aawm_adapter_runtime"
)
INIT_PATH = ADAPTER_PACKAGE_DIR / "__init__.py"
ADAPTER_CALLS_PATH = ADAPTER_PACKAGE_DIR / "anthropic_adapter_calls.py"
RESPONSE_UTILS_PATH = ADAPTER_PACKAGE_DIR / "response_utils.py"
ROUTING_COLLECT_TARGET = (
    REPO_ROOT
    / "tests/test_litellm/proxy/pass_through_endpoints/test_moonshot_alias_routing.py"
)
PASSTHROUGH_MODULE = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"
LLM_PASSTHROUGH_MODULE = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
)
RETRYABLE_STATUS_NAME = "PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES"
WATERMARK_HOOK_NAME = "maybe_apply_passthrough_watermark_response"


def _worktree_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else str(REPO_ROOT) + os.pathsep + existing
    )
    return env


def _run_fresh_interpreter(script: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=_worktree_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _fresh_interpreter_import(module_name: str) -> subprocess.CompletedProcess[str]:
    return _run_fresh_interpreter(f"import {module_name}")


def _module_level_importfroms(path: Path) -> list[ast.ImportFrom]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in tree.body if isinstance(node, ast.ImportFrom)]


def _importfrom_names(node: ast.ImportFrom) -> set[str]:
    return {alias.name for alias in node.names}


def _init_eagerly_imports_anthropic_adapter_calls() -> bool:
    for node in _module_level_importfroms(INIT_PATH):
        names = _importfrom_names(node)
        if "anthropic_adapter_calls" not in names:
            continue
        if node.level == 1 and node.module in {None, ""}:
            return True
        if node.level == 1 and node.module == "anthropic_adapter_calls":
            return True
        if node.module == "anthropic_adapter_calls":
            return True
    return False


def _adapter_calls_imports_retryable_status_at_module_load() -> bool:
    for node in _module_level_importfroms(ADAPTER_CALLS_PATH):
        if RETRYABLE_STATUS_NAME not in _importfrom_names(node):
            continue
        module = node.module or ""
        if module == PASSTHROUGH_MODULE or module.endswith(
            ".pass_through_endpoints.pass_through_endpoints"
        ):
            return True
        if node.level >= 1 and module in {"pass_through_endpoints", ""}:
            return True
    return False


def _passthrough_cycle_still_present() -> bool:
    return (
        _init_eagerly_imports_anthropic_adapter_calls()
        and _adapter_calls_imports_retryable_status_at_module_load()
    )


def _is_watermark_hook_import(node: ast.AST) -> bool:
    if not isinstance(node, ast.ImportFrom):
        return False
    if WATERMARK_HOOK_NAME not in _importfrom_names(node):
        return False
    module = node.module or ""
    return module.endswith("aawm_text_watermark.response_hooks") or module.endswith(
        "response_hooks"
    )


@pytest.mark.parametrize(
    "module_name",
    (LLM_PASSTHROUGH_MODULE, PASSTHROUGH_MODULE),
)
def test_rr093_fresh_interpreter_imports_passthrough_modules(module_name: str) -> None:
    result = _fresh_interpreter_import(module_name)
    assert result.returncode == 0, (
        f"fresh interpreter failed to import {module_name}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    ("first_module", "second_module"),
    (
        (LLM_PASSTHROUGH_MODULE, PASSTHROUGH_MODULE),
        (PASSTHROUGH_MODULE, LLM_PASSTHROUGH_MODULE),
    ),
    ids=("llm-then-passthrough", "passthrough-then-llm"),
)
def test_rr093_fresh_interpreter_imports_succeed_regardless_of_order(
    first_module: str,
    second_module: str,
) -> None:
    script = f"import {first_module}\nimport {second_module}\n"
    result = _run_fresh_interpreter(script)
    assert result.returncode == 0, (
        "fresh interpreter failed to import passthrough modules in order "
        f"{first_module} then {second_module}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_rr093_routing_test_module_collects() -> None:
    assert ROUTING_COLLECT_TARGET.is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROUTING_COLLECT_TARGET),
            "--collect-only",
            "-q",
        ],
        cwd=str(REPO_ROOT),
        env=_worktree_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, (
        "routing test collection failed (RR-093 import cycle)\n"
        f"{combined}"
    )
    assert "ERROR collecting" not in combined
    assert "ImportError" not in combined
    assert "partially initialized" not in combined


def test_rr093_adapter_package_does_not_form_passthrough_import_cycle() -> None:
    assert not _passthrough_cycle_still_present(), (
        "RR-093 cycle still present: aawm_adapter_runtime/__init__.py eagerly "
        "imports anthropic_adapter_calls and anthropic_adapter_calls module-load "
        f"imports {RETRYABLE_STATUS_NAME} from pass_through_endpoints. Break the "
        "cycle inside the adapter package (lazy package import, lazy constant "
        "import, or a tiny shared constant module already in the package)."
    )


def test_rr118_response_utils_keeps_module_scope_watermark_import_only() -> None:
    tree = ast.parse(RESPONSE_UTILS_PATH.read_text(encoding="utf-8"))
    module_scope = [
        node for node in tree.body if _is_watermark_hook_import(node)
    ]
    inner = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if child is node:
                continue
            if _is_watermark_hook_import(child):
                inner.append((node.name, child.lineno))
    assert len(module_scope) == 1, (
        "response_utils.py must keep exactly one module-scope import of "
        f"{WATERMARK_HOOK_NAME}"
    )
    assert inner == [], (
        "RR-118: remove the inner duplicate "
        f"{WATERMARK_HOOK_NAME} import from {inner}"
    )
