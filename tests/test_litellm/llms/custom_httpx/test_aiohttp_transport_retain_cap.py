"""RR-035: cross-loop session close is skipped; retained sessions are capped."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientSession

from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


@pytest.mark.asyncio
async def test_close_owned_replaced_session_skips_different_loop() -> None:
    session = MagicMock(spec=ClientSession)
    session.closed = False
    session._loop = object()  # not the running loop
    close = AsyncMock()
    session.close = close

    transport = LiteLLMAiohttpTransport(client=session, owns_session=True)
    await transport._close_owned_replaced_session(session)
    close.assert_not_called()


@pytest.mark.asyncio
async def test_get_valid_client_session_retains_on_loop_mismatch() -> None:
    import asyncio

    # Must be ClientSession-shaped so isinstance(client, ClientSession) is True.
    old = MagicMock(spec=ClientSession)
    old.closed = False
    old._loop = object()  # different loop

    new_session = MagicMock(spec=ClientSession)
    new_session.closed = False
    new_session._loop = asyncio.get_running_loop()

    transport = LiteLLMAiohttpTransport(client=old, owns_session=True)
    with patch.object(
        transport, "_create_replacement_client_session", return_value=new_session
    ), patch.object(
        transport, "_is_session_close_safe_on_current_loop", return_value=False
    ):
        got = await transport._get_valid_client_session_for_request()
    assert got is new_session
    assert old in transport._retained_replaced_sessions


def test_retained_sessions_cap_enforced() -> None:
    transport = LiteLLMAiohttpTransport(
        client=MagicMock(spec=ClientSession), owns_session=True
    )
    transport._MAX_RETAINED_REPLACED_SESSIONS = 3
    for _ in range(5):
        s = MagicMock(spec=ClientSession)
        s.closed = False
        s._loop = object()
        transport._retain_replaced_owned_session(s)
    assert len(transport._retained_replaced_sessions) <= 3
