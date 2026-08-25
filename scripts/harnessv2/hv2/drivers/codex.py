"""Codex interactive driver. Argv and forbid_flags live in YAML. Never -p/exec."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from hv2.envscrub import scrubbed_child_env
from hv2.errors import HarnessError, PlanError
from hv2.load_config import as_str_list, expand_string
from hv2.pane import (
    _CODEX_TOOL_PASS_TOKEN,
    _latest_prompt_echo_index,
    _pane_exact_pong,
    _pane_has_any,
    _pane_has_codex_idle_prompt,
    _pane_tool_command_pass,
)

_MODEL_ID_CONTINUE = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:+/"
)
_MODEL_CHROME_RE = re.compile(r"model:\s+(\S+)", re.IGNORECASE)
_LOADING_CHROME = "loading"


def _selected_needle_in_pane(token: str, text: str) -> bool:
    """True when *token* is in *text* without a longer model-id suffix."""

    if not token:
        return False
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return False
        end = index + len(token)
        if end < len(text) and text[end] in _MODEL_ID_CONTINUE:
            start = index + 1
            continue
        return True


def _latest_codex_model_chrome(text: str) -> str | None:
    """Return the latest Codex `model:` header token, if any.

    Codex 0.149 paints `model: loading` in the header while the footer
    already shows `{alias} default`. A bare `{model}` needle must not
    treat that footer as selected, or paste queues as a follow-up.
    """

    latest: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.replace("│", " ").replace("|", " ")
        match = _MODEL_CHROME_RE.search(line)
        if match:
            latest = match.group(1).strip().rstrip(".,;:")
    return latest or None


def _chrome_selects_model(chrome: str | None, model: str, selector: str) -> bool:
    if not chrome or not model:
        return False
    token = chrome.lower()
    if token == _LOADING_CHROME:
        return False
    wanted = {model.lower(), selector.lower()}
    return token in wanted


class CodexDriver:
    def __init__(self, config: Mapping[str, Any]) -> None:
        tuis = config.get("tuis") if isinstance(config.get("tuis"), dict) else {}
        spec = tuis.get("codex") if isinstance(tuis.get("codex"), dict) else {}
        if not spec:
            raise PlanError("tuis.codex is missing from tuis.yaml")
        self.config = config
        self.spec = spec
        self.forbid_flags = as_str_list(spec.get("forbid_flags"))
        self.forbid_tokens = as_str_list(spec.get("forbid_tokens")) or ["exec"]
        self._active_session: str | None = None
        self._active_model: str | None = None
        self._active_cwd: str | None = None

    def _context(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        lanes = self.spec.get("lanes") if isinstance(self.spec.get("lanes"), dict) else {}
        ctx = {
            "home": str(Path.home()),
            "session_dir": str(self.spec.get("session_dir") or "/tmp/hv2-codex-sessions"),
            "cwd": str(
                self._active_cwd
                or self.spec.get("cwd")
                or "/tmp/hv2-codex-workspace"
            ),
            "lane": str(lanes.get("alias") or "litellm-alpha"),
            "model": "",
            "selector": "",
            "pattern": "",
            "provider": str(lanes.get("alias") or "litellm-alpha"),
        }
        if extra:
            ctx.update({str(k): str(v) for k, v in extra.items()})
        return ctx

    def expand_argv(self, key: str, extra: Mapping[str, str] | None = None) -> list[str]:
        raw = as_str_list(self.spec.get(key))
        ctx = self._context(extra)
        argv = [expand_string(token, ctx) for token in raw]
        self.assert_no_print_flags(argv)
        return argv

    def assert_no_print_flags(self, argv: Sequence[str]) -> None:
        tokens = [str(item) for item in argv]
        joined = " ".join(tokens)
        for flag in (*self.forbid_flags, *self.forbid_tokens):
            if flag and flag in tokens:
                raise PlanError(
                    f"Codex argv contains forbidden flag {flag!r}; "
                    "interactive sessions must not use -p/--print or codex exec"
                )
        if "exec -p" in joined or "exec --print" in joined:
            raise PlanError(
                "Codex argv contains forbidden `exec -p`; "
                "interactive sessions must not use print/exec"
            )

    def child_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env_spec = self.spec.get("env") if isinstance(self.spec.get("env"), dict) else {}
        ctx = self._context()
        overlay = {
            str(key): expand_string(str(value), ctx) for key, value in env_spec.items()
        }
        if extra:
            overlay.update({str(k): str(v) for k, v in extra.items()})
        return scrubbed_child_env(self.config, overlay)

    def model_selector(self, model: str, *, lane: str | None = None) -> str:
        lanes = self.spec.get("lanes") if isinstance(self.spec.get("lanes"), dict) else {}
        chosen = lane or str(lanes.get("alias") or "litellm-alpha")
        template = str(self.spec.get("model_id_template") or "{model}")
        return expand_string(
            template, self._context({"lane": chosen, "model": model, "provider": chosen})
        )

    def detect_codex_version(self) -> str:
        configured = str(self.spec.get("version_min") or "0.142.5").strip() or "0.142.5"
        binary = shutil.which(str(self.spec.get("binary") or "codex"))
        if binary is None:
            return configured
        try:
            proc = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return configured
        text = " ".join(
            part.strip()
            for part in ((proc.stdout or ""), (proc.stderr or ""))
            if part and part.strip()
        )
        for token in text.replace(",", " ").split():
            token = token.strip()
            if token.lower().startswith("codex-cli/"):
                version = token.split("/", 1)[1].strip()
                if version:
                    return version
            if token.lower().startswith("codex/"):
                version = token.split("/", 1)[1].strip()
                if version:
                    return version
            if token[:1].isdigit() and all(
                char.isalnum() or char in "._-+" for char in token
            ):
                return token
        return configured

    def _identity_spec(self) -> dict[str, Any]:
        spec = self.spec.get("identity")
        return spec if isinstance(spec, dict) else {}

    def identity_overlay_payload(self, version: str | None = None) -> dict[str, Any]:
        resolved = str(version or self.detect_codex_version()).strip() or "0.142.5"
        identity = self._identity_spec()
        raw_headers = identity.get("headers") if isinstance(identity.get("headers"), dict) else {}
        headers = {
            str(key): str(value)
            for key, value in raw_headers.items()
            if key and value is not None
        }
        headers.setdefault("x-aawm-client", str(identity.get("client") or "Codex"))
        headers.setdefault(
            "x-aawm-client-name", str(identity.get("client_name") or "Codex")
        )
        headers["x-aawm-client-version"] = resolved
        headers.setdefault(
            "x-aawm-repository", str(identity.get("repository") or "litellm")
        )
        providers = as_str_list(identity.get("providers")) or [
            str((self.spec.get("lanes") or {}).get("alias") or "litellm-alpha")
        ]
        return {
            "headers": dict(headers),
            "providers": {name: {"headers": dict(headers)} for name in providers},
        }

    def identity_c_overrides(self, version: str | None = None) -> list[str]:
        payload = self.identity_overlay_payload(version=version)
        identity = self._identity_spec()
        lanes = self.spec.get("lanes") if isinstance(self.spec.get("lanes"), dict) else {}
        provider = str(lanes.get("alias") or "litellm-alpha")
        header_template = str(
            identity.get("header_override")
            or 'model_providers.{provider}.http_headers.{header}="{value}"'
        )
        provider_template = str(
            identity.get("model_provider_override") or 'model_provider="{lane}"'
        )
        overrides = [
            expand_string(
                provider_template, self._context({"lane": provider, "provider": provider})
            )
        ]
        headers = payload.get("headers") if isinstance(payload.get("headers"), dict) else {}
        for header, value in headers.items():
            overrides.append(
                expand_string(
                    header_template,
                    self._context(
                        {
                            "lane": provider,
                            "provider": provider,
                            "header": str(header),
                            "value": str(value),
                        }
                    ),
                )
            )
        return overrides

    def alias_session_dir(self, model: str) -> Path:
        parent = Path(str(self.spec.get("session_dir") or "/tmp/hv2-codex-sessions"))
        safe_model = model.replace("/", "-").replace(" ", "-")
        path = parent / f"hv2-{safe_model}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_workspace(self, session: str) -> Path:
        """Return a dedicated cwd so leftover hv2-codex sessions do not share jobs."""

        root = str(self.spec.get("cwd") or "/tmp/hv2-codex-workspace")
        if not session or session == self._default_session_name():
            path = Path(root)
        else:
            path = Path(f"{root}-{session}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def launch_argv(
        self, model: str, *, lane: str | None = None, cwd: str | None = None
    ) -> list[str]:
        selector = self.model_selector(model, lane=lane)
        lanes = self.spec.get("lanes") if isinstance(self.spec.get("lanes"), dict) else {}
        lane_s = lane or str(lanes.get("alias") or "litellm-alpha")
        chosen_cwd = str(
            cwd
            or self._active_cwd
            or self.spec.get("cwd")
            or "/tmp/hv2-codex-workspace"
        )
        extra = {
            "lane": lane_s,
            "model": model,
            "selector": selector,
            "provider": lane_s,
            "session_dir": str(self.alias_session_dir(model)),
            "cwd": chosen_cwd,
        }
        argv = self.expand_argv("argv_launch_model", extra)
        for override in self.identity_c_overrides():
            argv.extend(["-c", override])
        ctx = self._context(extra)
        for extra_override in as_str_list(self.spec.get("extra_c_overrides")):
            if extra_override:
                argv.extend(["-c", expand_string(extra_override, ctx)])
        self.assert_no_print_flags(argv)
        return argv

    def describe_session(self) -> dict[str, Any]:
        tmux = self.spec.get("tmux") if isinstance(self.spec.get("tmux"), dict) else {}
        return {
            "tui": "codex",
            "implemented": True,
            "binary": self.spec.get("binary"),
            "cwd": self.spec.get("cwd"),
            "session_dir": self.spec.get("session_dir"),
            "tmux_socket": tmux.get("socket"),
            "tmux_session": tmux.get("session"),
            "forbid_flags": list(self.forbid_flags),
            "select_model": dict(self._select_spec()),
        }

    def ensure_workspace(self) -> None:
        Path(str(self.spec.get("cwd") or "/tmp/hv2-codex-workspace")).mkdir(
            parents=True, exist_ok=True
        )
        if self._active_cwd:
            Path(self._active_cwd).mkdir(parents=True, exist_ok=True)
        Path(str(self.spec.get("session_dir") or "/tmp/hv2-codex-sessions")).mkdir(
            parents=True, exist_ok=True
        )

    def _tmux_cfg(self) -> dict[str, Any]:
        tmux = self.spec.get("tmux") if isinstance(self.spec.get("tmux"), dict) else {}
        return tmux

    def _select_spec(self) -> dict[str, Any]:
        spec = self.spec.get("select_model")
        return spec if isinstance(spec, dict) else {}

    def _tmux_float(self, key: str, default: float) -> float:
        raw = self._tmux_cfg().get(key)
        if raw is None:
            return default
        return float(raw)

    def _tmux_bin(self) -> str:
        tmux_bin = shutil.which(str(self._tmux_cfg().get("binary") or "tmux"))
        if tmux_bin is None:
            raise HarnessError("tmux is required for the Codex interactive driver")
        return tmux_bin

    def _tmux_socket(self) -> str:
        return str(self._tmux_cfg().get("socket") or "tmux37")

    def _default_session_name(self) -> str:
        return str(self._tmux_cfg().get("session") or "codex")

    def _session_name(self) -> str:
        return self._active_session or self._default_session_name()

    def _run_tmux(
        self,
        args: Sequence[str],
        *,
        timeout: int = 10,
        stdin_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._tmux_bin(), "-L", self._tmux_socket(), *[str(item) for item in args]],
            input=stdin_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def tmux_has_session(self, name: str | None = None) -> bool:
        session = name or self._session_name()
        try:
            proc = self._run_tmux(["has-session", "-t", session])
        except HarnessError:
            return False
        return proc.returncode == 0

    def _submit_keys(self) -> list[str]:
        keys = [token for token in as_str_list(self.spec.get("submit_keys")) if token]
        return keys or ["C-m"]

    def _submit_delay_seconds(self) -> float:
        raw = self.spec.get("submit_delay_seconds")
        if raw is None:
            raw = self._tmux_cfg().get("submit_delay_seconds")
        if raw is None:
            return 1.0
        try:
            delay = float(raw)
        except (TypeError, ValueError):
            return 1.0
        return delay if delay > 0 else 0.0

    def send_keys(self, text: str) -> dict[str, Any]:
        """Submit *text* to the active interactive tmux session. Not codex exec -p.

        Codex 0.149 maps Enter to composer newline. Dedicated harness
        sessions paste the prompt and submit with YAML ``submit_keys``
        (default ``C-m``) after YAML ``submit_delay_seconds`` so the
        composer can ingest the paste. Wait until the latest ``model:``
        header is the launched alias, not leftover footer
        ``{alias} default`` while the header still says ``loading``.
        Never send this into leftover operator panes.
        """

        self.assert_no_print_flags(["codex", text])
        model = str(self._active_model or "").strip()
        if model and not self._wait_until_model_selected(model):
            return {
                "ok": False,
                "returncode": 1,
                "stderr": (
                    "Codex model chrome is still loading; "
                    "refusing to queue a follow-up paste"
                ),
                "method": "paste-buffer",
                "submit_keys": self._submit_keys(),
                "submit_delay_seconds": self._submit_delay_seconds(),
            }
        session = self._session_name()
        payload = text if text.endswith("\n") else f"{text}\n"
        loaded = self._run_tmux(["load-buffer", "-"], stdin_text=payload)
        pasted = self._run_tmux(["paste-buffer", "-d", "-t", session])
        delay = self._submit_delay_seconds()
        if delay > 0:
            time.sleep(delay)
        submit_keys = self._submit_keys()
        submitted = self._run_tmux(
            ["send-keys", "-t", session, *submit_keys]
        )
        ok = (
            loaded.returncode == 0
            and pasted.returncode == 0
            and submitted.returncode == 0
        )
        return {
            "ok": ok,
            "returncode": submitted.returncode
            if ok
            else (loaded.returncode or pasted.returncode or submitted.returncode),
            "stderr": loaded.stderr or pasted.stderr or submitted.stderr,
            "method": "paste-buffer",
            "submit_keys": list(submit_keys),
            "submit_delay_seconds": delay,
        }

    def capture_pane(self) -> str:
        try:
            proc = self._run_tmux(
                ["capture-pane", "-pt", self._session_name(), "-S", "-200"]
            )
        except HarnessError:
            return ""
        return proc.stdout or ""

    def wait_for_pane(
        self,
        needle: str | Sequence[str],
        timeout_seconds: float | None = None,
        *,
        prompt: str | None = None,
        after_echo_index: int | None = None,
    ) -> bool:
        needles = [needle] if isinstance(needle, str) else [str(item) for item in needle]
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else self._tmux_float("wait_ready_seconds", 20)
        )
        interval = self._tmux_float("poll_interval_seconds", 1)
        deadline = time.time() + timeout
        while time.time() < deadline:
            pane = self.capture_pane()
            if _pane_has_any(
                pane, needles, prompt=prompt, after_echo_index=after_echo_index
            ):
                token_needles = [
                    item for item in needles if item == _CODEX_TOOL_PASS_TOKEN
                ]
                if token_needles:
                    if _pane_tool_command_pass(
                        pane,
                        prompt or "",
                        token_needles,
                        after_echo_index=after_echo_index,
                    ):
                        return True
                    other_needles = [
                        item
                        for item in needles
                        if item and item != _CODEX_TOOL_PASS_TOKEN
                    ]
                    if other_needles and _pane_has_any(
                        pane,
                        other_needles,
                        prompt=prompt,
                        after_echo_index=after_echo_index,
                    ):
                        return True
                    time.sleep(interval)
                    continue
                return True
            time.sleep(interval)
        return False

    def wait_until_idle(
        self,
        timeout_seconds: float | None = None,
        *,
        prompt: str | None = None,
        after_echo_index: int | None = None,
    ) -> bool:
        select = self._select_spec()
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else self._tmux_float("wait_idle_seconds", 90)
        )
        interval = self._tmux_float("poll_interval_seconds", 1)
        idle_needles = as_str_list(select.get("idle_needles")) or [">"]
        generic_idle_needles = [
            token for token in idle_needles if token.strip() != ">"
        ]
        busy_needles = as_str_list(select.get("busy_needles")) or [
            "Thinking",
            "Working (",
            "Streaming",
            "Waiting For Remaining",
            "all running jobs",
            "Background job",
        ]
        deadline = time.time() + timeout
        while time.time() < deadline:
            pane = self.capture_pane()
            if any(token in pane for token in busy_needles):
                time.sleep(interval)
                continue
            if _pane_has_codex_idle_prompt(
                pane, prompt, after_echo_index=after_echo_index
            ) or _pane_has_any(
                pane,
                generic_idle_needles,
                prompt=prompt,
                after_echo_index=after_echo_index,
            ):
                return True
            time.sleep(interval)
        return False

    def _session_env_pairs(self) -> list[str]:
        child = self.child_env()
        select = self._select_spec()
        keys = as_str_list(select.get("env_keys"))
        if not keys:
            keys = [
                "PATH",
                "HOME",
                "USER",
                "LOGNAME",
                "TERM",
                "COLORTERM",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "AAWM_HARNESS_USER_ID",
            ]
        pairs: list[str] = []
        for key in keys:
            value = child.get(key)
            if value:
                pairs.append(f"{key}={value}")
        for key, value in child.items():
            if key.startswith("CODEX_") or key.startswith("OPENAI_"):
                token = f"{key}={value}"
                if token not in pairs:
                    pairs.append(token)
        return pairs

    def _trust_prompt_needles(self) -> list[str]:
        return [
            token
            for token in as_str_list(self._select_spec().get("trust_prompt_needles"))
            if token
        ]

    def _pane_has_trust_prompt(self, pane: str) -> bool:
        needles = self._trust_prompt_needles()
        return bool(needles) and any(token in pane for token in needles)

    def _pane_selected_without_trust(self, pane: str, model: str) -> bool:
        if not model or self._pane_has_trust_prompt(pane):
            return False
        return self.pane_has_selector(model, pane)

    def _send_trust_prompt_enter(self) -> bool:
        session = self._session_name()
        if not session or session == self._default_session_name():
            return False
        proc = self._run_tmux(["send-keys", "-t", session, "Enter"])
        return proc.returncode == 0

    def _accept_trust_prompt_if_present(self, model: str = "") -> bool:
        """Accept the first-run directory-trust nux in a dedicated harness session.

        Codex 0.149 paints ``model: loading`` before the nux. ``wait_trust_seconds``
        is too short for that splash; keep polling until the selected model is
        visible or ``wait_ready_seconds`` elapses. Prompt-free panes that already
        show the selected model return immediately and never send Enter.
        Never send-keys the leftover operator ``codex`` pane.
        """

        needles = self._trust_prompt_needles()
        if not needles:
            return False
        ready_timeout = self._tmux_float("wait_ready_seconds", 20)
        trust_timeout = self._tmux_float("wait_trust_seconds", 3)
        interval = max(self._tmux_float("poll_interval_seconds", 1), 0.05)
        deadline = time.time() + max(ready_timeout, trust_timeout, 0.0)
        pane = self.capture_pane()
        while not self._pane_has_trust_prompt(pane):
            if self._pane_selected_without_trust(pane, model):
                return False
            if time.time() >= deadline:
                return False
            time.sleep(interval)
            pane = self.capture_pane()
        if not self._send_trust_prompt_enter():
            return False
        clear_deadline = time.time() + max(ready_timeout, 0.0)
        while time.time() < clear_deadline:
            pane = self.capture_pane()
            if not self._pane_has_trust_prompt(pane):
                return True
            time.sleep(interval)
        return False

    def pane_has_selector(self, model: str, pane: str | None = None) -> bool:
        selector = self.model_selector(model)
        text = pane if pane is not None else self.capture_pane()
        chrome = _latest_codex_model_chrome(text)
        if chrome is not None:
            return _chrome_selects_model(chrome, model, selector)
        needles = [
            expand_string(token, self._context({"selector": selector, "model": model}))
            for token in as_str_list(self._select_spec().get("selected_needles"))
        ]
        if not needles:
            needles = [selector, model]
        return any(_selected_needle_in_pane(token, text) for token in needles)

    def _wait_until_model_selected(
        self, model: str, timeout_seconds: float | None = None
    ) -> bool:
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else self._tmux_float("wait_ready_seconds", 20)
        )
        interval = self._tmux_float("poll_interval_seconds", 1)
        deadline = time.time() + max(timeout, 0.0)
        while True:
            pane = self.capture_pane()
            if self._pane_has_trust_prompt(pane):
                self._send_trust_prompt_enter()
                pane = self.capture_pane()
            if self.pane_has_selector(model, pane):
                return True
            if timeout <= 0 or time.time() >= deadline:
                return self.pane_has_selector(model)
            time.sleep(interval)

    def ensure_session(
        self,
        model: str,
        *,
        tools: bool = True,
        child_agents: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Launch a dedicated interactive Codex tmux session for *model*.

        Does not reuse leftover operator ``codex`` panes. Never uses
        ``-p`` / ``--print`` / ``codex exec``.
        """

        _ = child_agents
        select = self._select_spec()
        if select.get("reuse_operator_session") is True:
            raise PlanError(
                "tuis.codex.select_model.reuse_operator_session is true; "
                "harness v2 must not send-keys leftover operator Codex panes"
            )
        prefix = str(self._tmux_cfg().get("harness_session_prefix") or "hv2-codex")
        safe_model = model.replace("/", "-").replace(" ", "-")
        session = f"{prefix}-{safe_model}-{os.getpid()}"
        operator = self._default_session_name()
        if session == operator:
            raise PlanError(
                "refusing to overwrite the operator Codex session "
                f"{operator}; set tmux.harness_session_prefix"
            )
        cwd = str(self.session_workspace(session))
        self._active_cwd = cwd
        self.ensure_workspace()
        argv = self.launch_argv(model, cwd=cwd)
        if not tools:
            argv.extend(as_str_list(self.spec.get("argv_no_tools")))
        self.assert_no_print_flags(argv)
        if self.tmux_has_session(session):
            self._run_tmux(["kill-session", "-t", session])
        tmux_args = ["new-session", "-d", "-s", session, "-c", cwd]
        for pair in self._session_env_pairs():
            tmux_args.extend(["-e", pair])
        tmux_args.extend(argv)
        proc = self._run_tmux(tmux_args)
        if proc.returncode != 0:
            raise HarnessError(
                f"tmux new-session {session} failed: "
                f"{(proc.stderr or proc.stdout or '').strip()}"
            )
        self._active_session = session
        self._active_model = model
        ready_needles = as_str_list(select.get("ready_needles")) or ["codex"]
        ready = self.wait_for_pane(
            ready_needles,
            timeout_seconds=self._tmux_float("wait_ready_seconds", 20),
        )
        self._accept_trust_prompt_if_present(model)
        selector = self.model_selector(model)
        selected = self._wait_until_model_selected(
            model,
            timeout_seconds=self._tmux_float("wait_ready_seconds", 20),
        )
        mcp_needles = as_str_list(select.get("mcp_ready_needles"))
        mcp_ready = True
        if mcp_needles:
            mcp_ready = self.wait_for_pane(
                mcp_needles,
                timeout_seconds=self._tmux_float("wait_mcp_seconds", 30),
            )
        pane = self.capture_pane()
        rejected = [
            token
            for token in as_str_list(select.get("reject_needles"))
            if token and token in pane
        ]
        return {
            "ok": bool(ready and selected and mcp_ready and not rejected),
            "session": session,
            "argv": argv,
            "selector": selector,
            "ready": ready,
            "selected": selected,
            "mcp_ready": mcp_ready,
            "rejected": rejected,
            "pane_preview": pane[-800:],
            "staged_agents": None,
        }

    def send_prompt_and_wait(
        self,
        prompt: str,
        *,
        reply_needles: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        select = self._select_spec()
        needles = [str(item) for item in (reply_needles or []) if item]
        if not needles:
            needles = as_str_list(select.get("pass_needles")) + as_str_list(
                select.get("error_needles")
            )
        sent_prompt = prompt.strip()
        pre_pane = self.capture_pane()
        pre_echo = _latest_prompt_echo_index(pre_pane, sent_prompt)
        sent = self.send_keys(sent_prompt)
        replied = False
        if needles:
            replied = self.wait_for_pane(
                needles,
                timeout_seconds=self._tmux_float("wait_reply_seconds", 180),
                prompt=sent_prompt,
                after_echo_index=pre_echo,
            )
        pane = self.capture_pane()
        if not replied and _pane_exact_pong(
            pane, sent_prompt, after_echo_index=pre_echo
        ):
            replied = True
        idle = False
        if replied:
            idle = self.wait_until_idle(
                prompt=sent_prompt, after_echo_index=pre_echo
            )
        pane = self.capture_pane()
        return {
            "ok": bool(sent.get("ok") and replied and idle),
            "send": sent,
            "idle": idle,
            "replied": replied,
            "pane": pane,
            "after_echo_index": pre_echo,
        }

    def close_session(self) -> None:
        session = self._active_session
        if not session:
            return
        if session == self._default_session_name():
            self._active_session = None
            self._active_model = None
            self._active_cwd = None
            return
        self._run_tmux(["kill-session", "-t", session])
        self._active_session = None
        self._active_model = None
        self._active_cwd = None
