"""Grok interactive driver. grokla env + inspect-derived alpha /grok/v1.

Dedicated tmux sessions only. Never send-keys leftover operator
``grok`` / ``groklt`` panes. Never aims at :4000 / :4001.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from hv2.envscrub import scrubbed_child_env
from hv2.errors import HarnessError, PlanError, ProtectedTargetError
from hv2.grokla import (
    GROKLA_WRAPPER_NAME,
    GROKLT_WRAPPER_NAME,
    assert_grokla_proxy_url,
    grok_binary,
    grokla_proxy_url_from_resolved,
    resolve_grokla_proxy_url,
)
from hv2.load_config import as_str_list, expand_string
from hv2.pane import _latest_prompt_echo_index, _pane_exact_pong, _pane_has_any


class GrokDriver:
    def __init__(self, config: Mapping[str, Any]) -> None:
        tuis = config.get("tuis") if isinstance(config.get("tuis"), dict) else {}
        spec = tuis.get("grok") if isinstance(tuis.get("grok"), dict) else {}
        if not spec:
            raise PlanError("tuis.grok is missing from tuis.yaml")
        self.config = config
        self.spec = spec
        self.forbid_flags = as_str_list(spec.get("forbid_flags"))
        self.forbid_tokens = as_str_list(spec.get("forbid_tokens")) or [
            GROKLT_WRAPPER_NAME,
            "grokl",
        ]
        self._active_session: str | None = None
        self._active_model: str | None = None
        self._proxy_url: str | None = None

    def bind_resolved(self, resolved: Any) -> str:
        """Pin GROK_CLI_CHAT_PROXY_BASE_URL to the inspected alpha /grok/v1 URL."""

        self._proxy_url = grokla_proxy_url_from_resolved(resolved, self.config)
        return self._proxy_url

    def proxy_url(self) -> str:
        if self._proxy_url:
            return assert_grokla_proxy_url(self._proxy_url, self.config)
        return resolve_grokla_proxy_url(self.config)

    def _context(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        lanes = self.spec.get("lanes") if isinstance(self.spec.get("lanes"), dict) else {}
        ctx = {
            "home": str(Path.home()),
            "session_dir": str(self.spec.get("session_dir") or "/tmp/hv2-grok-sessions"),
            "cwd": str(self.spec.get("cwd") or "/tmp/hv2-grok-workspace"),
            "lane": str(lanes.get("alias") or "litellm-alpha"),
            "model": "",
            "selector": "",
            "pattern": "",
            "wrapper": GROKLA_WRAPPER_NAME,
        }
        if extra:
            ctx.update({str(k): str(v) for k, v in extra.items()})
        return ctx

    def expand_argv(self, key: str, extra: Mapping[str, str] | None = None) -> list[str]:
        raw = as_str_list(self.spec.get(key))
        ctx = self._context(extra)
        argv = [expand_string(token, ctx) for token in raw]
        if argv and argv[0] in {"grok", GROKLA_WRAPPER_NAME}:
            argv[0] = grok_binary()
        self.assert_no_print_flags(argv)
        return argv

    def assert_no_print_flags(self, argv: Sequence[str]) -> None:
        tokens = [str(item) for item in argv]
        joined = " ".join(tokens)
        for flag in (*self.forbid_flags, *self.forbid_tokens):
            if flag and flag in tokens:
                raise PlanError(
                    f"Grok argv contains forbidden token {flag!r}; "
                    "harness v2 must not use groklt or leftover operator panes"
                )
        lowered = joined.lower()
        if GROKLT_WRAPPER_NAME in lowered or "grokl " in lowered + " ":
            raise PlanError(
                "Grok argv must not invoke groklt; grokla targets litellm-alpha only"
            )
        for token in (":4000", ":4001", "aawm-litellm", "litellm-dev"):
            if token in lowered:
                raise ProtectedTargetError(
                    f"Grok argv refuses {token}: grokla never targets "
                    "aawm-litellm / litellm-dev / :4000 / :4001"
                )

    def child_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env_spec = self.spec.get("env") if isinstance(self.spec.get("env"), dict) else {}
        ctx = self._context()
        overlay = {
            str(key): expand_string(str(value), ctx) for key, value in env_spec.items()
        }
        overlay["GROK_CLI_CHAT_PROXY_BASE_URL"] = self.proxy_url()
        overlay.setdefault("GROK_DISABLE_UPDATE_CHECK", "1")
        overlay.setdefault("GROK_SANDBOX", "workspace")
        overlay.setdefault("GROK_SUBAGENTS", "1")
        if extra:
            overlay.update({str(k): str(v) for k, v in extra.items()})
        env = scrubbed_child_env(self.config, overlay)
        env["GROK_CLI_CHAT_PROXY_BASE_URL"] = self.proxy_url()
        return env

    def model_selector(self, model: str, *, lane: str | None = None) -> str:
        lanes = self.spec.get("lanes") if isinstance(self.spec.get("lanes"), dict) else {}
        chosen = lane or str(lanes.get("alias") or "litellm-alpha")
        template = str(self.spec.get("model_id_template") or "{model}")
        return expand_string(template, self._context({"lane": chosen, "model": model}))

    def launch_argv(self, model: str, *, lane: str | None = None) -> list[str]:
        selector = self.model_selector(model, lane=lane)
        extra = {
            "model": model,
            "selector": selector,
            "session_dir": str(self.alias_session_dir(model)),
            "cwd": str(self.spec.get("cwd") or "/tmp/hv2-grok-workspace"),
        }
        argv = self.expand_argv("argv_launch_model", extra)
        self.assert_no_print_flags(argv)
        return argv

    def alias_session_dir(self, model: str) -> Path:
        root = str(self.spec.get("session_dir") or "/tmp/hv2-grok-sessions")
        safe = model.replace("/", "-").replace(" ", "-")
        path = Path(f"{root}/hv2-{safe}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def describe_session(self) -> dict[str, Any]:
        tmux = self.spec.get("tmux") if isinstance(self.spec.get("tmux"), dict) else {}
        return {
            "tui": "grok",
            "implemented": True,
            "wrapper": GROKLA_WRAPPER_NAME,
            "binary": self.spec.get("binary"),
            "cwd": self.spec.get("cwd"),
            "session_dir": self.spec.get("session_dir"),
            "tmux_socket": tmux.get("socket"),
            "tmux_session": tmux.get("session"),
            "forbid_flags": list(self.forbid_flags),
            "proxy_url": self._proxy_url,
            "select_model": dict(self._select_spec()),
        }

    def ensure_workspace(self) -> None:
        Path(str(self.spec.get("cwd") or "/tmp/hv2-grok-workspace")).mkdir(
            parents=True, exist_ok=True
        )
        Path(str(self.spec.get("session_dir") or "/tmp/hv2-grok-sessions")).mkdir(
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
            raise HarnessError("tmux is required for the Grok interactive driver")
        return tmux_bin

    def _tmux_socket(self) -> str:
        return str(self._tmux_cfg().get("socket") or "tmux37")

    def _default_session_name(self) -> str:
        return str(self._tmux_cfg().get("session") or "grok")

    def _session_name(self) -> str:
        return self._active_session or self._default_session_name()

    def _tmux_target(self, name: str | None = None) -> str:
        """Exact tmux session target. Dots in `grok-4.6` are pane paths.

        ``=name`` is not enough: tmux still splits ``grok-4.6`` as
        ``session:window.pane``. A trailing colon pins the session.
        """
        session = name or self._session_name()
        if not session or session.startswith("%"):
            return session
        if session.startswith("="):
            return session if session.endswith(":") else f"{session}:"
        return f"={session}:"

    def _with_exact_tmux_targets(self, args: Sequence[str]) -> list[str]:
        rewritten: list[str] = []
        pending_target = False
        for item in args:
            token = str(item)
            if pending_target:
                rewritten.append(
                    token if token.startswith("-") else self._tmux_target(token)
                )
                pending_target = False
                continue
            if token in {"-t", "-pt"}:
                pending_target = True
            rewritten.append(token)
        return rewritten

    def _run_tmux(
        self,
        args: Sequence[str],
        *,
        timeout: int = 10,
        stdin_text: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                self._tmux_bin(),
                "-L",
                self._tmux_socket(),
                *self._with_exact_tmux_targets(args),
            ],
            input=stdin_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=dict(env) if env is not None else None,
        )

    def tmux_has_session(self, name: str | None = None) -> bool:
        session = name or self._session_name()
        try:
            proc = self._run_tmux(["has-session", "-t", session])
        except HarnessError:
            return False
        return proc.returncode == 0

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
                "GROK_CLI_CHAT_PROXY_BASE_URL",
                "GROK_DISABLE_UPDATE_CHECK",
                "GROK_SANDBOX",
                "GROK_SUBAGENTS",
                "AAWM_GROK_REAL_BIN",
            ]
        pairs: list[str] = []
        for key in keys:
            value = child.get(key)
            if value:
                pairs.append(f"{key}={value}")
        for key, value in child.items():
            if key.startswith("GROK_") or key.startswith("XAI_"):
                token = f"{key}={value}"
                if token not in pairs:
                    pairs.append(token)
        return pairs

    def pane_has_selector(self, model: str, pane: str | None = None) -> bool:
        selector = self.model_selector(model)
        text = pane if pane is not None else self.capture_pane()
        needles = [
            expand_string(token, self._context({"selector": selector, "model": model}))
            for token in as_str_list(self._select_spec().get("selected_needles"))
        ]
        if not needles:
            needles = [selector, model]
        return any(token and token in text for token in needles)

    def send_keys(self, text: str) -> dict[str, Any]:
        """Submit *text* to the dedicated grokla tmux session."""

        self.assert_no_print_flags(["grok", text])
        session = self._session_name()
        operator = self._default_session_name()
        if session == operator:
            raise PlanError(
                "refusing to send-keys leftover operator grok/groklt pane "
                f"{operator}; dedicated hv2-grok sessions only"
            )
        payload = text if text.endswith("\n") else f"{text}\n"
        submit_keys = as_str_list(self.spec.get("submit_keys")) or ["C-m"]
        delay = float(self.spec.get("submit_delay_seconds") or 1.0)
        if "\n" in text.strip("\n"):
            loaded = self._run_tmux(["load-buffer", "-"], stdin_text=payload)
            pasted = self._run_tmux(["paste-buffer", "-d", "-t", session])
            if delay > 0:
                time.sleep(delay)
            submitted = self._run_tmux(["send-keys", "-t", session, *submit_keys])
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
            }
        proc = self._run_tmux(["send-keys", "-t", session, text, *submit_keys])
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr": proc.stderr,
            "method": "send-keys",
        }

    def capture_pane(self) -> str:
        try:
            proc = self._run_tmux(
                ["capture-pane", "-pt", self._session_name(), "-S", "-1000"]
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
        sent_prompt = prompt or ""
        while time.time() < deadline:
            pane = self.capture_pane()
            if _pane_has_any(
                pane, needles, prompt=prompt, after_echo_index=after_echo_index
            ):
                return True
            if sent_prompt and _pane_exact_pong(
                pane, sent_prompt, after_echo_index=after_echo_index
            ):
                return True
            time.sleep(interval)
        return False

    def wait_until_idle(self, timeout_seconds: float | None = None) -> bool:
        select = self._select_spec()
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else self._tmux_float("wait_idle_seconds", 90)
        )
        interval = self._tmux_float("poll_interval_seconds", 1)
        idle_needles = as_str_list(select.get("idle_needles")) or [">"]
        busy_needles = as_str_list(select.get("busy_needles")) or [
            "Thinking",
            "Working",
            "Streaming",
        ]
        deadline = time.time() + timeout
        while time.time() < deadline:
            pane = self.capture_pane()
            if any(token in pane for token in busy_needles):
                time.sleep(interval)
                continue
            if any(token in pane for token in idle_needles):
                return True
            time.sleep(interval)
        return False

    def ensure_session(
        self,
        model: str,
        *,
        tools: bool = True,
        child_agents: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Launch a dedicated interactive grokla tmux session for *model*."""

        _ = child_agents
        select = self._select_spec()
        if select.get("reuse_operator_session") is True:
            raise PlanError(
                "tuis.grok.select_model.reuse_operator_session is true; "
                "harness v2 must not send-keys leftover grok/groklt panes"
            )
        self.ensure_workspace()
        argv = self.launch_argv(model)
        if not tools:
            argv.extend(as_str_list(self.spec.get("argv_no_tools")))
        self.assert_no_print_flags(argv)
        prefix = str(self._tmux_cfg().get("harness_session_prefix") or "hv2-grok")
        safe_model = model.replace("/", "-").replace(" ", "-").replace(".", "-")
        session = f"{prefix}-{safe_model}-{os.getpid()}"
        operator = self._default_session_name()
        if session == operator:
            raise PlanError(
                "refusing to overwrite the operator Grok session "
                f"{operator}; set tmux.harness_session_prefix"
            )
        if self.tmux_has_session(session):
            self._run_tmux(["kill-session", "-t", session])
        cwd = str(self.spec.get("cwd") or "/tmp/hv2-grok-workspace")
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
        ready_needles = as_str_list(select.get("ready_needles")) or ["grok", "Grok"]
        ready = self.wait_for_pane(
            ready_needles,
            timeout_seconds=self._tmux_float("wait_ready_seconds", 25),
        )
        selector = self.model_selector(model)
        selected_needles = [
            expand_string(token, self._context({"selector": selector, "model": model}))
            for token in as_str_list(select.get("selected_needles"))
        ] or [selector, model]
        selected = self.wait_for_pane(
            selected_needles,
            timeout_seconds=self._tmux_float("wait_ready_seconds", 25),
        )
        pane = self.capture_pane()
        rejected = [
            token
            for token in as_str_list(select.get("reject_needles"))
            if token and token in pane
        ]
        return {
            "ok": bool(ready and selected and not rejected),
            "session": session,
            "argv": argv,
            "selector": selector,
            "ready": ready,
            "selected": selected,
            "mcp_ready": True,
            "rejected": rejected,
            "pane_preview": pane[-800:],
            "staged_agents": None,
            "proxy_url": self.proxy_url(),
            "wrapper": GROKLA_WRAPPER_NAME,
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
                timeout_seconds=self._tmux_float("wait_reply_seconds", 420),
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
            idle = self.wait_until_idle()
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
            return
        self._run_tmux(["kill-session", "-t", session])
        self._active_session = None
        self._active_model = None
