import aiohttp
import threading
from unittest.mock import patch

import pytest

from litellm.llms.custom_httpx.aiohttp_transport import (
    get_litellm_aiohttp_session_attribution,
)
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking


@pytest.mark.asyncio
async def test_sanity_presidio_session_reuse_main_thread():
    """
    SANITY CHECK:
    Verify that Presidio guardrail reuses sessions in the main thread.
    This ensures we don't break existing session pooling functionality.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )

    session_creations = 0
    original_init = aiohttp.ClientSession.__init__

    def mocked_init(self, *args, **kwargs):
        nonlocal session_creations
        session_creations += 1
        original_init(self, *args, **kwargs)

    with patch.object(
        aiohttp.ClientSession, "__init__", side_effect=mocked_init, autospec=True
    ):
        for _ in range(10):
            async with presidio._get_session_iterator() as _session:
                pass

        # Expected: Only 1 session created for all 10 calls.
        assert session_creations == 1

    await presidio._close_http_session()


@pytest.mark.asyncio
async def test_bug_presidio_session_explosion_background_thread_causes_latency():
    """
    BUG REPRODUCTION:
    Verify that background threads (like logging hooks) REUSE sessions.
    Previously, each call in a background loop created a NEW ephemeral session,
    leading to socket exhaustion and the reported 97s latency spike.
    """

    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )

    # Force the code to think it's in a background thread
    presidio._main_thread_id = threading.get_ident() + 1

    session_creations = 0
    original_init = aiohttp.ClientSession.__init__

    def mocked_init(self, *args, **kwargs):
        nonlocal session_creations
        session_creations += 1
        original_init(self, *args, **kwargs)

    with patch.object(
        aiohttp.ClientSession, "__init__", side_effect=mocked_init, autospec=True
    ):
        for _ in range(10):
            async with presidio._get_session_iterator() as _session:
                pass

        # FIX VERIFICATION: Should now be 1 session (reused) instead of 10.
        assert session_creations == 1

    await presidio._close_http_session()


@pytest.mark.asyncio
async def test_presidio_session_iterator_marks_main_and_loop_sessions():
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )

    async with presidio._get_session_iterator() as main_session:
        main_attribution = get_litellm_aiohttp_session_attribution(main_session)
    assert main_attribution is not None
    assert main_attribution["owner_kind"] == "presidio"
    assert (
        main_attribution["creation_site"]
        == "presidio._get_session_iterator:main-thread"
    )
    assert main_attribution["litellm_owns_session"] is True

    presidio._main_thread_id = threading.get_ident() + 1
    async with presidio._get_session_iterator() as loop_session:
        loop_attribution = get_litellm_aiohttp_session_attribution(loop_session)

    assert loop_attribution is not None
    assert loop_attribution["owner_kind"] == "presidio"
    assert (
        loop_attribution["creation_site"]
        == "presidio._get_session_iterator:loop-session"
    )
    assert loop_attribution["litellm_owns_session"] is True

    await presidio._close_http_session()
