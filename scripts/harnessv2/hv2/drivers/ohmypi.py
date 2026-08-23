"""Ohmypi interactive driver. Argv and forbid_flags live in YAML. Never -p."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from hv2.envscrub import scrubbed_child_env
from hv2.errors import HarnessError, PlanError
from hv2.load_config import as_str_list, config_timeouts, expand_string
from hv2.pane import _latest_prompt_echo_index, _pane_exact_pong, _pane_has_any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise PlanError("PyYAML is required to stage Ohmypi identity overlays") from exc

_MODEL_ID_CONTINUE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:+/")


def _selected_needle_in_pane(token: str, text: str) -> bool:
    """True when *token* is in *text* without a longer model-id suffix.

    ``AAWM alias / model work`` must not match ``AAWM alias / model work-other``.
    """

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


class OhmypiDriver:
    def __init__(self, config: Mapping[str, Any]) -> None:
        tuis = config.get("tuis") if isinstance(config.get("tuis"), dict) else {}
        spec = tuis.get("ohmypi") if isinstance(tuis.get("ohmypi"), dict) else {}
        if not spec:
            raise PlanError("tuis.ohmypi is missing from tuis.yaml")
        self.config = config
        self.spec = spec
        self.forbid_flags = as_str_list(spec.get("forbid_flags"))
        self._active_session: str | None = None

    def _context(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        ctx = {
            "home": str(Path.home()),
            "session_dir": str(self.spec.get("session_dir") or "/tmp/omp-alpha-sessions"),
            "cwd": str(self.spec.get("cwd") or "/tmp/omp-alpha-workspace"),
            "lane": str((self.spec.get("lanes") or {}).get("alias") or "litellm-alpha-passthrough"),
            "model": "",
            "selector": "",
            "pattern": "",
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
        for flag in self.forbid_flags:
            if flag and flag in argv:
                raise PlanError(
                    f"Ohmypi argv contains forbidden flag {flag!r}; "
                    "interactive sessions must not use -p/--print"
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
        chosen = lane or str(lanes.get("alias") or "litellm-alpha-passthrough")
        template = str(self.spec.get("model_id_template") or "{lane}/{model}")
        return expand_string(template, self._context({"lane": chosen, "model": model}))

    def detect_ohmypi_version(self) -> str:
        configured = str(self.spec.get("version_min") or "17.3.8").strip() or "17.3.8"
        binary = shutil.which(str(self.spec.get("binary") or "omp"))
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
            if token.lower().startswith("omp/"):
                version = token.split("/", 1)[1].strip()
                if version:
                    return version
            if token[:1].isdigit() and all(
                char.isalnum() or char in "._-+" for char in token
            ):
                return token
        return configured

    def identity_overlay_payload(self, version: str | None = None) -> dict[str, Any]:
        resolved = str(version or self.detect_ohmypi_version()).strip() or "17.3.8"
        headers = {
            "x-aawm-client": "Oh My Pi",
            "x-aawm-client-name": "omp",
            "x-aawm-client-version": resolved,
            "x-aawm-repository": "litellm",
            "langfuse_trace_name": "omp",
        }
        return {
            "providers": {
                "litellm-alpha": {"headers": dict(headers)},
                "litellm-alpha-passthrough": {"headers": dict(headers)},
            }
        }

    def identity_overlay_path(self) -> Path:
        session_dir = Path(str(self.spec.get("session_dir") or "/tmp/omp-alpha-sessions"))
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / "hv2-ohmypi-identity.yml"

    def alias_session_dir(self, model: str) -> Path:
        parent = Path(str(self.spec.get("session_dir") or "/tmp/omp-alpha-sessions"))
        safe_model = model.replace("/", "-").replace(" ", "-")
        path = parent / f"hv2-{safe_model}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_identity_overlay(self, version: str | None = None) -> Path:
        path = self.identity_overlay_path()
        path.write_text(
            yaml.safe_dump(self.identity_overlay_payload(version=version), sort_keys=False),
            encoding="utf-8",
        )
        return path

    def launch_argv(self, model: str, *, lane: str | None = None) -> list[str]:
        selector = self.model_selector(model, lane=lane)
        lane_s, _, model_s = selector.partition("/")
        if not model_s:
            lane_s = str(
                (self.spec.get("lanes") or {}).get("alias") or "litellm-alpha-passthrough"
            )
            model_s = model
        argv = self.expand_argv(
            "argv_launch_model",
            {
                "lane": lane_s,
                "model": model_s,
                "selector": selector,
                "session_dir": str(self.alias_session_dir(model)),
            },
        )
        overlay = self.write_identity_overlay()
        if "--config" not in argv:
            argv.extend(["--config", str(overlay)])
        self.assert_no_print_flags(argv)
        return argv

    def describe_session(self) -> dict[str, Any]:
        tmux = self.spec.get("tmux") if isinstance(self.spec.get("tmux"), dict) else {}
        return {
            "tui": "ohmypi",
            "implemented": True,
            "binary": self.spec.get("binary"),
            "wrapper": self.spec.get("wrapper"),
            "cwd": self.spec.get("cwd"),
            "session_dir": self.spec.get("session_dir"),
            "tmux_socket": tmux.get("socket"),
            "tmux_session": tmux.get("session"),
            "forbid_flags": list(self.forbid_flags),
            "spawn_prefix": self.spec.get("spawn_prefix"),
            "select_model": dict(self._select_spec()),
        }

    def ensure_workspace(self) -> None:
        Path(str(self.spec.get("cwd") or "/tmp/omp-alpha-workspace")).mkdir(
            parents=True, exist_ok=True
        )
        Path(str(self.spec.get("session_dir") or "/tmp/omp-alpha-sessions")).mkdir(
            parents=True, exist_ok=True
        )

    def stage_orchestration_agents(self) -> dict[str, Any]:
        """Copy harness child profiles into the Ohmypi project agents dir.

        Ohmypi `task` spawn accepts `agent=` profile names from
        `{cwd}/.omp/agents`, not LiteLLM catalog ids. Staging here keeps
        operator `~/.omp/agent/agents` untouched.
        """

        ctx = self._context()
        cwd = Path(ctx["cwd"])
        raw_src = str(self.spec.get("bundled_agents_dir") or "")
        src = Path(
            expand_string(raw_src, ctx)
            if raw_src
            else (Path(__file__).resolve().parents[1] / "config" / "ohmypi-agents")
        )
        raw_dest = str(self.spec.get("project_agents_dir") or "")
        dest = Path(
            expand_string(raw_dest, ctx) if raw_dest else (cwd / ".omp" / "agents")
        )
        dest.mkdir(parents=True, exist_ok=True)
        names = as_str_list(self.spec.get("orchestration_child_agents")) or [
            "basic",
            "work",
            "expert",
            "sota",
        ]
        written: list[str] = []
        missing: list[str] = []
        for name in names:
            source = src / f"{name}.md"
            if not source.is_file():
                missing.append(str(source))
                continue
            target = dest / f"{name}.md"
            shutil.copy2(source, target)
            written.append(str(target))
        return {
            "ok": bool(written) and not missing,
            "src": str(src),
            "dest": str(dest),
            "written": written,
            "missing": missing,
        }

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
            raise HarnessError("tmux is required for the Ohmypi interactive driver")
        return tmux_bin

    def _tmux_socket(self) -> str:
        return str(self._tmux_cfg().get("socket") or "tmux37")

    def _default_session_name(self) -> str:
        return str(self._tmux_cfg().get("session") or "omp-alpha-test")

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

    def catalog_json(self) -> dict[str, Any]:
        argv = self.expand_argv("catalog_json_argv")
        env = self.child_env()
        timeout = config_timeouts(self.config)["tui_seconds"]
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.spec.get("cwd") or "/tmp"),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise HarnessError(f"Ohmypi binary not found: {argv[0]}") from exc
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "ok": proc.returncode == 0,
        }

    def catalog_find(self, pattern: str) -> dict[str, Any]:
        argv = self.expand_argv("catalog_find_argv", {"pattern": pattern})
        env = self.child_env()
        timeout = config_timeouts(self.config)["tui_seconds"]
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.spec.get("cwd") or "/tmp"),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise HarnessError(f"Ohmypi binary not found: {argv[0]}") from exc
        return {
            "argv": argv,
            "pattern": pattern,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "ok": proc.returncode == 0,
        }

    def tmux_has_session(self, name: str | None = None) -> bool:
        session = name or self._session_name()
        try:
            proc = self._run_tmux(["has-session", "-t", session])
        except HarnessError:
            return False
        return proc.returncode == 0

    def send_keys(self, text: str) -> dict[str, Any]:
        """Submit *text* to the active interactive tmux session. Not omp -p.

        Single-line prompts use ``send-keys … Enter``. Multiline prompts
        (orchestration fanout) go through ``load-buffer`` + ``paste-buffer``
        so Ohmypi does not keep a ``[Paste #N]`` draft unsubmitted.
        """

        self.assert_no_print_flags(["omp", text])
        session = self._session_name()
        payload = text if text.endswith("\n") else f"{text}\n"
        if "\n" in text.strip("\n"):
            loaded = self._run_tmux(["load-buffer", "-"], stdin_text=payload)
            pasted = self._run_tmux(["paste-buffer", "-d", "-t", session])
            submitted = self._run_tmux(["send-keys", "-t", session, "Enter"])
            ok = (
                loaded.returncode == 0
                and pasted.returncode == 0
                and submitted.returncode == 0
            )
            return {
                "ok": ok,
                "returncode": submitted.returncode if ok else (loaded.returncode or pasted.returncode or submitted.returncode),
                "stderr": loaded.stderr or pasted.stderr or submitted.stderr,
                "method": "paste-buffer",
            }
        proc = self._run_tmux(["send-keys", "-t", session, text, "Enter"])
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr": proc.stderr,
            "method": "send-keys",
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
        sent_prompt = prompt or ""
        while time.time() < deadline:
            pane = self.capture_pane()
            if _pane_has_any(
                pane, needles, prompt=prompt, after_echo_index=after_echo_index
            ):
                return True
            # Recap is wait-only. Exact PONG after a prompt echo newer than
            # the pre-send watermark still completes the wait when recap
            # never paints.
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
        idle_needles = as_str_list(select.get("idle_needles")) or [
            "╰─",
            "π  >",
            "Default model:",
        ]
        busy_needles = as_str_list(select.get("busy_needles")) or [
            "Thinking",
            "Working",
            "Streaming",
            "Waiting For Remaining",
            "all running jobs",
            "Background job",
        ]
        # True in-flight tokens keep the pane busy even when the idle
        # footer is already painted. Launch splash leftover is not a
        # busy needle (see tuis.yaml) and must not block idle.
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
                "PI_CONFIG_FILES",
                "AAWM_HARNESS_USER_ID",
            ]
        pairs: list[str] = []
        for key in keys:
            value = child.get(key)
            if value:
                pairs.append(f"{key}={value}")
        for key, value in child.items():
            if key.startswith("PI_") or key.startswith("OMP_"):
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
            needles = [selector, f"Default model: {selector}"]
        return any(_selected_needle_in_pane(token, text) for token in needles)

    def ensure_session(self, model: str, *, tools: bool = True) -> dict[str, Any]:
        """Launch a dedicated interactive Ohmypi tmux session for *model*.

        Does not reuse ``omp-alpha-test`` so leftover Claude aliases cannot
        steal the turn. Never uses ``-p`` / ``--print``.
        """

        select = self._select_spec()
        if select.get("reuse_operator_session") is True:
            raise PlanError(
                "tuis.ohmypi.select_model.reuse_operator_session is true; "
                "harness v2 must not send-keys leftover omp-alpha-test panes"
            )
        self.ensure_workspace()
        staged_agents: dict[str, Any] | None = None
        if tools:
            staged_agents = self.stage_orchestration_agents()
        argv = self.launch_argv(model)
        if not tools:
            argv.extend(as_str_list(self.spec.get("argv_no_tools")))
        self.assert_no_print_flags(argv)
        prefix = str(self._tmux_cfg().get("harness_session_prefix") or "hv2-ohmypi")
        safe_model = model.replace("/", "-").replace(" ", "-")
        session = f"{prefix}-{safe_model}-{os.getpid()}"
        operator = self._default_session_name()
        if session == operator:
            raise PlanError(
                "refusing to overwrite the operator Ohmypi session "
                f"{operator}; set tmux.harness_session_prefix"
            )
        if self.tmux_has_session(session):
            self._run_tmux(["kill-session", "-t", session])
        cwd = str(self.spec.get("cwd") or "/tmp/omp-alpha-workspace")
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
        ready_needles = as_str_list(select.get("ready_needles")) or ["π"]
        ready = self.wait_for_pane(
            ready_needles,
            timeout_seconds=self._tmux_float("wait_ready_seconds", 20),
        )
        selector = self.model_selector(model)
        selected_needles = [
            expand_string(token, self._context({"selector": selector, "model": model}))
            for token in as_str_list(select.get("selected_needles"))
        ]
        if not selected_needles:
            selected_needles = [selector, f"Default model: {selector}"]
        selected_timeout = self._tmux_float("wait_ready_seconds", 20)
        selected = self.wait_for_pane(
            selected_needles,
            timeout_seconds=selected_timeout,
        )
        mcp_needles = as_str_list(select.get("mcp_ready_needles"))
        mcp_ready = True
        if mcp_needles:
            mcp_ready = self.wait_for_pane(
                mcp_needles,
                timeout_seconds=self._tmux_float("wait_mcp_seconds", 30),
            )
        if not selected:
            # Alias chrome can paint after MCP connect on long model ids.
            selected = self.wait_for_pane(
                selected_needles,
                timeout_seconds=selected_timeout,
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
            "staged_agents": staged_agents,
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
        # Recap is wait-only. Exact PONG after a prompt echo newer than the
        # pre-send watermark still completes the model wait when recap never
        # paints. A restored complete echo+PONG turn is not this send.
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
        }

    def close_session(self) -> None:
        session = self._active_session
        if not session:
            return
        if session == self._default_session_name():
            self._active_session = None
            return
        self._run_tmux(["kill-session", "-t", session])
        self._active_session = None
