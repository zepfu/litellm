"""Wave 2 structural pin: the alias candidate retry loop lives in the package.

Extends the ``test_rr054_structural_extraction.py`` AST-ownership discipline to
the Wave-2 candidate-loop extraction
(``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``):

- the loop body (``handle_alias_route``) is defined in
  ``aawm_alias_routing/candidate_loop.py`` and must NOT be re-defined as a
  function body in the god-module
- ``_handle_auto_agent_alias_route`` remains on the god-module (39 test files
  depend on the name) but is now a THIN façade that delegates to
  ``candidate_loop.handle_alias_route``
- the typed seam bundle (``AliasRouteServices``) is defined in
  ``aawm_alias_routing/interfaces.py`` and must not be re-defined on the
  god-module

Write-only surface: this file. No production edits.
"""

from __future__ import annotations

import ast
from pathlib import Path

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

PACKAGE_DIR = Path(package.__file__).resolve().parent
GOD_PATH = Path(lpe.__file__).resolve()
CANDIDATE_LOOP_PATH = PACKAGE_DIR / "candidate_loop.py"
INTERFACES_PATH = PACKAGE_DIR / "interfaces.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.AST:
    return ast.parse(_read(path), filename=str(path))


def _top_level_function_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _function_node(tree: ast.AST, name: str) -> ast.AST | None:
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _call_attr_names(fn_node: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
            continue
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.add(".".join(reversed(parts)))
    return names


def test_candidate_loop_body_is_defined_in_package() -> None:
    """``handle_alias_route`` is a real function body in ``candidate_loop.py``."""
    assert CANDIDATE_LOOP_PATH.is_file(), CANDIDATE_LOOP_PATH
    loop_source = _read(CANDIDATE_LOOP_PATH)
    tree = _parse(CANDIDATE_LOOP_PATH)
    assert "handle_alias_route" in _top_level_function_names(tree), "candidate_loop.py must define handle_alias_route"
    # Substance markers proving the R3-1 widened-lock body lives here.
    for marker in (
        "async def handle_alias_route",
        "candidate_probe_lock",
        "probe_lock.release()",
        "CooldownPublicationPlan",
    ):
        assert marker in loop_source, f"candidate_loop.py missing loop-body marker {marker!r}"


def test_god_module_does_not_define_loop_body() -> None:
    """The loop body must NOT be re-defined as a function in the god-module."""
    god_tree = _parse(GOD_PATH)
    god_fns = _top_level_function_names(god_tree)
    assert "handle_alias_route" not in god_fns, (
        "god-module re-defines the candidate loop body (handle_alias_route); "
        "it must delegate to aawm_alias_routing.candidate_loop instead"
    )


def test_god_facade_is_thin_delegate_to_candidate_loop() -> None:
    """``_handle_auto_agent_alias_route`` stays on the god-module but delegates."""
    god_source = _read(GOD_PATH)
    assert (
        "async def _handle_auto_agent_alias_route(" in god_source
    ), "the legacy loop entrypoint name must be preserved (39 test files depend on it)"
    fn = _function_node(_parse(GOD_PATH), "_handle_auto_agent_alias_route")
    assert fn is not None
    calls = _call_attr_names(fn)
    assert "_aawm_alias_candidate_loop.handle_alias_route" in calls, (
        "_handle_auto_agent_alias_route must delegate to " f"candidate_loop.handle_alias_route; calls={sorted(calls)}"
    )


def test_alias_route_services_defined_in_interfaces_not_god() -> None:
    """``AliasRouteServices`` is owned by ``interfaces.py``, not the god-module."""
    assert INTERFACES_PATH.is_file(), INTERFACES_PATH
    iface_tree = _parse(INTERFACES_PATH)
    iface_names = {
        node.name
        for node in iface_tree.body  # type: ignore[union-attr]
        if isinstance(node, ast.ClassDef)
    }
    assert "AliasRouteServices" in iface_names, "interfaces.py must define AliasRouteServices"
    god_tree = _parse(GOD_PATH)
    god_classes = {
        node.name
        for node in god_tree.body  # type: ignore[union-attr]
        if isinstance(node, ast.ClassDef)
    }
    assert (
        "AliasRouteServices" not in god_classes
    ), "god-module must not redefine AliasRouteServices; it imports it from interfaces"
