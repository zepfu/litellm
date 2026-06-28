"""
Utility functions for cleaning up async HTTP clients to prevent resource leaks.
"""
import asyncio
import inspect

MAX_CLEANUP_FAILURE_LOGS = 3


async def _close_cached_client_entry(handler, label: str, _safe_close) -> int:
    """Close one cache entry or retained evicted client-like value."""
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    if isinstance(handler, BaseLLMAIOHTTPHandler) and hasattr(handler, "close"):
        await _safe_close(handler.close, f"{label} BaseLLMAIOHTTPHandler.close")
        return 1
    elif hasattr(handler, "client"):
        client = handler.client
        if hasattr(client, "_transport") and hasattr(client._transport, "aclose"):
            await _safe_close(
                client._transport.aclose,
                f"{label} client._transport.aclose",
            )
        if hasattr(client, "aclose"):
            await _safe_close(client.aclose, f"{label} client.aclose")
        return 1
    elif hasattr(handler, "aclose"):
        await _safe_close(handler.aclose, f"{label} generic.aclose")
        return 1
    elif hasattr(handler, "close"):
        await _safe_close(handler.close, f"{label} generic.close")
        return 1
    return 0


async def close_litellm_async_clients():
    """
    Close all cached async HTTP clients to prevent resource leaks.

    This function iterates through all cached clients in litellm's in-memory cache,
    closes any retained evicted client-like entries held by ``LLMClientCache``,
    and closes any aiohttp client sessions that are still open. Also closes the
    global base_llm_aiohttp_handler instance (issue #12443).
    """
    # Import here to avoid circular import
    import litellm
    from litellm._logging import verbose_logger
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    cache_dict = getattr(litellm.in_memory_llm_clients_cache, "cache_dict", {})
    close_count = 0
    failures: list[str] = []

    async def _safe_close(close_fn, label: str) -> None:
        if not callable(close_fn):
            return
        try:
            result = close_fn()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            failures.append(f"{label}: {type(e).__name__}: {e}")

    def _log_summary():
        if failures:
            sample = failures[:MAX_CLEANUP_FAILURE_LOGS]
            if len(failures) > MAX_CLEANUP_FAILURE_LOGS:
                sample.append(
                    f"... {len(failures) - MAX_CLEANUP_FAILURE_LOGS} more"
                )
            verbose_logger.warning(
                "close_litellm_async_clients completed with %s failures: %s",
                len(failures),
                "; ".join(sample),
            )
        elif close_count:
            verbose_logger.debug(
                "close_litellm_async_clients cleaned %s async client entries", close_count
            )
        else:
            verbose_logger.debug(
                "close_litellm_async_clients no async client entries required cleanup"
            )

    for key, handler in cache_dict.items():
        close_count += await _close_cached_client_entry(
            handler,
            f"cache[{key}]",
            _safe_close,
        )

    client_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    retained = getattr(client_cache, "_evicted_clients_retained", None)
    if retained:
        for index, handler in enumerate(list(retained)):
            close_count += await _close_cached_client_entry(
                handler,
                f"retained_evicted[{index}]",
                _safe_close,
            )
        retained.clear()

    if hasattr(litellm, "base_llm_aiohttp_handler"):
        base_handler = getattr(litellm, "base_llm_aiohttp_handler", None)
        if isinstance(base_handler, BaseLLMAIOHTTPHandler) and hasattr(
            base_handler, "close"
        ):
            await _safe_close(
                base_handler.close,
                "global base_llm_aiohttp_handler.close",
            )
            close_count += 1

    _log_summary()


def register_async_client_cleanup():
    """
    Register the async client cleanup function to run at exit.

    This ensures that all async HTTP clients are properly closed when the program exits.
    """
    import atexit

    def cleanup_wrapper():
        """
        Cleanup wrapper that creates a fresh event loop for atexit cleanup.

        At exit time, the main event loop is often already closed. Creating a new
        event loop ensures cleanup runs successfully (fixes issue #12443).
        """
        try:
            # Always create a fresh event loop at exit time
            # Don't use get_event_loop() - it may be closed or unavailable
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(close_litellm_async_clients())
            finally:
                # Clean up the loop we created
                loop.close()
        except Exception:
            # Silently ignore errors during cleanup to avoid exit handler failures
            from litellm._logging import verbose_logger

            verbose_logger.warning(
                "Failed to close async clients during atexit cleanup.",
                exc_info=True,
            )

    atexit.register(cleanup_wrapper)
