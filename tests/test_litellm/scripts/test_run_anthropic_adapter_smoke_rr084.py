"""RR-084: normalize URLError/timeout failures in adapter smoke _post_json.

_post_json previously only caught urllib.error.HTTPError. Connection refused
and socket timeouts raised as raw low-level exceptions instead of the script's
RuntimeError diagnostic style.
"""

from __future__ import annotations

import importlib.util
import io
import socket
import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_smoke.py"


def _load_module():
    name = "run_anthropic_adapter_smoke_rr084"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke():
    return _load_module()


def _headers() -> dict[str, str]:
    return {"content-type": "application/json"}


def _payload() -> dict[str, Any]:
    return {"model": "gpt-5.4", "messages": []}


def test_should_normalize_urlerror_connection_refused_to_runtime_error(smoke) -> None:
    url = "http://127.0.0.1:9/anthropic/v1/messages"
    err = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    with patch.object(smoke.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(RuntimeError) as exc_info:
            smoke._post_json(url, _headers(), _payload(), timeout=5.0)

    message = str(exc_info.value)
    assert "adapter smoke request failed connecting to" in message
    assert url in message
    assert "Connection refused" in message
    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)


def test_should_normalize_bare_timeout_error_to_runtime_error(smoke) -> None:
    url = "http://127.0.0.1:4001/anthropic/v1/messages"
    timeout = 1.5

    with patch.object(smoke.urllib.request, "urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(RuntimeError) as exc_info:
            smoke._post_json(url, _headers(), _payload(), timeout=timeout)

    message = str(exc_info.value)
    assert "adapter smoke request timed out" in message
    assert f"{timeout}s" in message
    assert url in message
    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_should_normalize_urlerror_wrapping_socket_timeout_to_runtime_error(smoke) -> None:
    url = "http://127.0.0.1:4001/anthropic/v1/messages"
    timeout = 2.0
    # urllib often wraps socket.timeout / TimeoutError as URLError.reason
    sock_timeout = socket.timeout("timed out")
    err = urllib.error.URLError(sock_timeout)

    with patch.object(smoke.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(RuntimeError) as exc_info:
            smoke._post_json(url, _headers(), _payload(), timeout=timeout)

    message = str(exc_info.value)
    assert "adapter smoke request timed out" in message
    assert f"{timeout}s" in message
    assert url in message
    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)


def test_should_preserve_httperror_details_in_runtime_error(smoke) -> None:
    url = "http://127.0.0.1:4001/anthropic/v1/messages"
    body = b'{"error":{"message":"upstream failed","type":"api_error"}}'
    http_err = urllib.error.HTTPError(
        url=url,
        code=502,
        msg="Bad Gateway",
        hdrs=None,
        fp=io.BytesIO(body),
    )

    with patch.object(smoke.urllib.request, "urlopen", side_effect=http_err):
        with pytest.raises(RuntimeError) as exc_info:
            smoke._post_json(url, _headers(), _payload(), timeout=5.0)

    message = str(exc_info.value)
    assert "adapter smoke request failed with HTTP 502" in message
    assert "upstream failed" in message
    assert isinstance(exc_info.value.__cause__, urllib.error.HTTPError)
    # Ensure we did not demote HTTPError into the connect-failure message path.
    assert "failed connecting" not in message
    assert "timed out" not in message


def test_should_return_status_and_parsed_body_on_success(smoke) -> None:
    url = "http://127.0.0.1:4001/anthropic/v1/messages"
    response = MagicMock()
    response.status = 200
    response.read.return_value = b'{"type":"message","model":"gpt-5.4"}'
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch.object(smoke.urllib.request, "urlopen", return_value=response):
        status_code, parsed = smoke._post_json(url, _headers(), _payload(), timeout=5.0)

    assert status_code == 200
    assert parsed == {"type": "message", "model": "gpt-5.4"}
