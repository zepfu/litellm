import asyncio
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import redis.exceptions
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache


def _build_redis_cache(sync_client: Any) -> RedisCache:
    with patch(
        "litellm._redis.get_redis_client",
        return_value=sync_client,
    ), patch(
        "litellm._redis.get_redis_connection_pool",
        return_value=MagicMock(),
    ), patch.object(
        RedisCache,
        "_setup_health_pings",
    ):
        return RedisCache()


def _single_dual_cache_read_record(
    caplog: pytest.LogCaptureFixture,
    *,
    error_class: str,
    raise_on_error: bool,
) -> Any:
    records = [
        record
        for record in caplog.records
        if record.getMessage() == "LiteLLM DualCache Redis dependency failure"
    ]
    assert len(records) == 1
    assert len(
        [record for record in caplog.records if record.levelno >= logging.ERROR]
    ) == 1
    record = records[0]
    assert record.exc_info is None
    assert record.pathname.endswith("dual_cache.py")
    assert "Traceback (most recent call last)" not in record.getMessage()
    assert record.operation == "read"
    assert record.error_class == error_class
    assert record.dependency == "redis"
    assert record.raise_on_error is raise_on_error
    assert record.disposition == ("raise" if raise_on_error else "degraded_miss")
    return record


def _single_unexpected_traceback_record(
    caplog: pytest.LogCaptureFixture,
    error: BaseException,
) -> Any:
    records = [
        record for record in caplog.records if record.levelno >= logging.ERROR
    ]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() != "LiteLLM DualCache Redis dependency failure"
    assert "Traceback (most recent call last)" in record.getMessage()
    assert str(error) in record.getMessage()
    return record


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_coalesces_concurrent_redis_reads():
    dual_cache = DualCache(
        redis_cache=MagicMock(spec=RedisCache), default_redis_batch_cache_expiry=10
    )
    keys = ["shared_a", "shared_b"]
    start_gate = asyncio.Event()

    async def _mock_async_batch_get_cache(key_list, parent_otel_span=None):
        await asyncio.sleep(0.05)
        return {k: None for k in key_list}

    with patch.object(
        dual_cache.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(side_effect=_mock_async_batch_get_cache),
    ) as mock_async_batch_get_cache:

        async def worker():
            await start_gate.wait()
            return await dual_cache.async_batch_get_cache(keys=keys)

        tasks = [asyncio.create_task(worker()) for _ in range(50)]
        start_gate.set()
        await asyncio.gather(*tasks)

        assert mock_async_batch_get_cache.call_count == 1


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_rolls_back_redis_reservation_on_error():
    dual_cache = DualCache(
        redis_cache=MagicMock(spec=RedisCache), default_redis_batch_cache_expiry=10
    )
    keys = ["shared_a", "shared_b"]

    with patch.object(
        dual_cache.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(side_effect=RuntimeError("redis unavailable")),
    ) as mock_async_batch_get_cache:
        first_result = await dual_cache.async_batch_get_cache(keys=keys)
        second_result = await dual_cache.async_batch_get_cache(keys=keys)

        assert first_result is None
        assert second_result is None
        assert mock_async_batch_get_cache.call_count == 2
        assert "shared_a" not in dual_cache.last_redis_batch_access_time
        assert "shared_b" not in dual_cache.last_redis_batch_access_time


@pytest.mark.parametrize(
    ("error", "error_class"),
    [
        pytest.param(
            redis.exceptions.ConnectionError("redis connection failed"),
            "connection",
            id="redis-connection",
        ),
        pytest.param(
            TimeoutError("builtin timeout"),
            "timeout",
            id="builtin-timeout",
        ),
        pytest.param(
            redis.exceptions.ClusterDownError("redis cluster down"),
            "availability",
            id="cluster-down",
        ),
        pytest.param(
            redis.exceptions.MasterDownError("redis master down"),
            "availability",
            id="master-down",
        ),
        pytest.param(
            redis.exceptions.TryAgainError("redis try again"),
            "availability",
            id="try-again",
        ),
    ],
)
@pytest.mark.parametrize("raise_on_error", [False, True])
def test_dual_cache_sync_real_redis_read_failure_is_bounded(
    caplog, error, error_class, raise_on_error
):
    sync_client = MagicMock()
    sync_client.get.side_effect = error
    redis_cache = _build_redis_cache(sync_client)
    service_logger = MagicMock()
    service_logger.async_service_failure_hook = AsyncMock()
    redis_cache.service_logger_obj = service_logger
    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,
    )

    with caplog.at_level(logging.ERROR):
        if raise_on_error:
            with pytest.raises(type(error)) as exc_info:
                dual_cache.get_cache("dual-cache-read", raise_on_error=True)
            assert exc_info.value is error
        else:
            assert dual_cache.get_cache("dual-cache-read") is None

    record = _single_dual_cache_read_record(
        caplog,
        error_class=error_class,
        raise_on_error=raise_on_error,
    )
    assert str(error) not in record.getMessage()
    assert not any("Got exception from REDIS" in item.getMessage() for item in caplog.records)
    service_logger.service_failure_hook.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "error_class"),
    [
        pytest.param(
            redis.exceptions.TimeoutError("redis timeout"),
            "timeout",
            id="redis-timeout",
        ),
        pytest.param(
            ConnectionError("builtin connection failed"),
            "connection",
            id="builtin-connection",
        ),
        pytest.param(
            redis.exceptions.ClusterDownError("redis cluster down"),
            "availability",
            id="cluster-down",
        ),
        pytest.param(
            redis.exceptions.MasterDownError("redis master down"),
            "availability",
            id="master-down",
        ),
        pytest.param(
            redis.exceptions.TryAgainError("redis try again"),
            "availability",
            id="try-again",
        ),
    ],
)
@pytest.mark.parametrize("raise_on_error", [False, True])
async def test_dual_cache_async_real_redis_read_failure_is_bounded(
    caplog, error, error_class, raise_on_error
):
    async_client = AsyncMock()
    async_client.get.side_effect = error
    redis_cache = _build_redis_cache(MagicMock())
    service_logger = MagicMock()
    service_logger.async_service_failure_hook = AsyncMock()
    redis_cache.service_logger_obj = service_logger
    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,
    )

    with caplog.at_level(logging.ERROR):
        with patch.object(redis_cache, "init_async_client", return_value=async_client):
            if raise_on_error:
                with pytest.raises(type(error)) as exc_info:
                    await dual_cache.async_get_cache(
                        "dual-cache-read",
                        raise_on_error=True,
                    )
                assert exc_info.value is error
            else:
                assert await dual_cache.async_get_cache("dual-cache-read") is None

    record = _single_dual_cache_read_record(
        caplog,
        error_class=error_class,
        raise_on_error=raise_on_error,
    )
    assert str(error) not in record.getMessage()
    assert not any("Got exception from REDIS" in item.getMessage() for item in caplog.records)
    service_logger.async_service_failure_hook.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("unexpected cache wiring failure"), id="runtime"),
        pytest.param(redis.exceptions.RedisError("generic redis failure"), id="redis"),
        pytest.param(redis.exceptions.ResponseError("redis response failure"), id="response"),
        pytest.param(redis.exceptions.DataError("redis data failure"), id="data"),
        pytest.param(
            redis.exceptions.InvalidResponse("redis protocol failure"),
            id="invalid-response",
        ),
        pytest.param(
            redis.exceptions.AuthenticationError("redis auth failure"),
            id="authentication",
        ),
        pytest.param(
            redis.exceptions.AuthorizationError("redis authorization failure"),
            id="authorization",
        ),
        pytest.param(ValueError("invalid redis configuration"), id="configuration"),
    ],
)
def test_dual_cache_sync_real_redis_nonavailability_failure_keeps_traceback(
    caplog, error
):
    sync_client = MagicMock()
    sync_client.get.side_effect = error
    redis_cache = _build_redis_cache(sync_client)
    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(type(error)) as exc_info:
            dual_cache.get_cache("dual-cache-read", raise_on_error=True)

    assert exc_info.value is error
    record = _single_unexpected_traceback_record(caplog, error)
    assert record.pathname.endswith("dual_cache.py")


@pytest.mark.asyncio
async def test_dual_cache_async_real_redis_unexpected_read_error_keeps_traceback(caplog):
    error = RuntimeError("unexpected cache wiring failure")
    async_client = AsyncMock()
    async_client.get.side_effect = error
    redis_cache = _build_redis_cache(MagicMock())
    service_logger = MagicMock()
    service_logger.async_service_failure_hook = AsyncMock()
    redis_cache.service_logger_obj = service_logger
    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,
    )

    with caplog.at_level(logging.ERROR):
        with patch.object(redis_cache, "init_async_client", return_value=async_client):
            with pytest.raises(RuntimeError, match="unexpected cache wiring failure"):
                await dual_cache.async_get_cache(
                    "dual-cache-read",
                    raise_on_error=True,
                )

    record = _single_unexpected_traceback_record(caplog, error)
    assert record.pathname.endswith("dual_cache.py")
    service_logger.async_service_failure_hook.assert_not_awaited()


def test_dual_cache_sync_unexpected_redis_error_degrades_with_one_traceback(caplog):
    error = redis.exceptions.RedisError("unexpected redis read failure")
    sync_client = MagicMock()
    sync_client.get.side_effect = error
    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=_build_redis_cache(sync_client),
    )

    with caplog.at_level(logging.ERROR):
        assert dual_cache.get_cache("dual-cache-read") is None

    record = _single_unexpected_traceback_record(caplog, error)
    assert record.pathname.endswith("redis_cache.py")


@pytest.mark.asyncio
async def test_dual_cache_async_unexpected_redis_error_degrades_with_one_traceback(
    caplog,
):
    error = RuntimeError("unexpected async redis read failure")
    async_client = AsyncMock()
    async_client.get.side_effect = error
    redis_cache = _build_redis_cache(MagicMock())
    service_logger = MagicMock()
    service_logger.async_service_failure_hook = AsyncMock()
    redis_cache.service_logger_obj = service_logger
    dual_cache = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_cache,
    )

    with caplog.at_level(logging.ERROR):
        with patch.object(redis_cache, "init_async_client", return_value=async_client):
            assert await dual_cache.async_get_cache("dual-cache-read") is None
            await asyncio.sleep(0)

    record = _single_unexpected_traceback_record(caplog, error)
    assert record.pathname.endswith("redis_cache.py")
    service_logger.async_service_failure_hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_dual_cache_async_set_cache_injects_default_in_memory_ttl():
    """
    Test that async_set_cache injects default_in_memory_ttl into kwargs
    when no explicit ttl is provided, matching the sync set_cache behavior.

    Regression test for: async_set_cache was missing the TTL injection that
    sync set_cache has, causing InMemoryCache to use its own default_ttl (600s)
    instead of DualCache's default_in_memory_ttl.
    """
    in_memory_cache = InMemoryCache(default_ttl=600)
    dual_cache = DualCache(
        in_memory_cache=in_memory_cache,
        default_in_memory_ttl=60,
    )

    before = time.time()
    await dual_cache.async_set_cache(key="test_key", value="test_value")
    after = time.time()

    # The TTL stored should reflect default_in_memory_ttl (60s), not
    # InMemoryCache's default_ttl (600s)
    expiry = in_memory_cache.ttl_dict["test_key"]
    assert expiry >= before + 60
    assert expiry <= after + 60


@pytest.mark.asyncio
async def test_dual_cache_async_set_cache_respects_explicit_ttl():
    """
    Test that async_set_cache does NOT override an explicitly provided ttl.
    """
    in_memory_cache = InMemoryCache(default_ttl=600)
    dual_cache = DualCache(
        in_memory_cache=in_memory_cache,
        default_in_memory_ttl=60,
    )

    before = time.time()
    await dual_cache.async_set_cache(key="test_key", value="test_value", ttl=30)
    after = time.time()

    # The explicit ttl=30 should be used, not default_in_memory_ttl (60)
    expiry = in_memory_cache.ttl_dict["test_key"]
    assert expiry >= before + 30
    assert expiry <= after + 30


@pytest.mark.asyncio
async def test_dual_cache_async_set_cache_pipeline_injects_default_in_memory_ttl():
    """
    Test that async_set_cache_pipeline injects default_in_memory_ttl into kwargs
    when no explicit ttl is provided.
    """
    in_memory_cache = InMemoryCache(default_ttl=600)
    dual_cache = DualCache(
        in_memory_cache=in_memory_cache,
        default_in_memory_ttl=60,
    )

    cache_list = [("key_a", "value_a"), ("key_b", "value_b")]

    before = time.time()
    await dual_cache.async_set_cache_pipeline(cache_list=cache_list)
    after = time.time()

    for key in ["key_a", "key_b"]:
        expiry = in_memory_cache.ttl_dict[key]
        assert expiry >= before + 60
        assert expiry <= after + 60


@pytest.mark.asyncio
async def test_dual_cache_sync_and_async_set_cache_use_same_ttl():
    """
    Test that sync set_cache and async async_set_cache produce the same TTL
    when no explicit ttl is provided, ensuring parity between the two paths.
    """
    in_memory_sync = InMemoryCache(default_ttl=600)
    dual_cache_sync = DualCache(
        in_memory_cache=in_memory_sync,
        default_in_memory_ttl=60,
    )

    in_memory_async = InMemoryCache(default_ttl=600)
    dual_cache_async = DualCache(
        in_memory_cache=in_memory_async,
        default_in_memory_ttl=60,
    )

    dual_cache_sync.set_cache(key="test_key", value="test_value")
    await dual_cache_async.async_set_cache(key="test_key", value="test_value")

    sync_expiry = in_memory_sync.ttl_dict["test_key"]
    async_expiry = in_memory_async.ttl_dict["test_key"]

    # Both should use default_in_memory_ttl=60, so their expiry times
    # should be within a small tolerance of each other
    assert abs(sync_expiry - async_expiry) < 1.0
