"""Focused tests for RR-059 proxy callback list reset hazards."""

from __future__ import annotations

import os
import tempfile
from typing import Any, List
from unittest.mock import patch

import pytest
import yaml

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import ProxyConfig


class _MarkerSuccessLogger(CustomLogger):
    def __init__(self, name: str = "marker"):
        super().__init__()
        self.name = name


class _MarkerGeneralLogger(CustomLogger):
    def __init__(self, name: str = "general"):
        super().__init__()
        self.name = name


def _reset_callback_state() -> None:
    litellm.logging_callback_manager._reset_all_callbacks()


def _count_instances(callback_list: List[Any], cls: type) -> int:
    return sum(isinstance(cb, cls) for cb in callback_list)


@pytest.fixture(autouse=True)
def _clean_callbacks():
    _reset_callback_state()
    yield
    _reset_callback_state()


def test_success_callback_reset_preserves_general_callbacks_registrations():
    """General `callbacks` ownership must survive success_callback list reset."""
    proxy_config = ProxyConfig()
    general_logger = _MarkerGeneralLogger(name="from-callbacks")
    success_only = _MarkerSuccessLogger(name="from-success")
    stale = _MarkerSuccessLogger(name="stale-should-go")

    litellm.callbacks = [general_logger]
    litellm.logging_callback_manager.add_litellm_success_callback(general_logger)
    litellm.logging_callback_manager.add_litellm_async_success_callback(general_logger)
    litellm.logging_callback_manager.add_litellm_success_callback(stale)
    litellm.logging_callback_manager.add_litellm_async_success_callback(stale)

    # Apply the same reset used by the success_callback config branch, then
    # re-register only the success-only logger (as that branch would).
    proxy_config._reset_event_callback_lists("success")
    litellm.logging_callback_manager.add_litellm_success_callback(success_only)
    litellm.logging_callback_manager.add_litellm_async_success_callback(success_only)

    assert general_logger in litellm.callbacks
    assert _count_instances(litellm.success_callback, _MarkerGeneralLogger) == 1
    assert _count_instances(litellm._async_success_callback, _MarkerGeneralLogger) == 1
    assert _count_instances(litellm.success_callback, _MarkerSuccessLogger) == 1
    assert _count_instances(litellm._async_success_callback, _MarkerSuccessLogger) == 1
    assert all(
        getattr(cb, "name", None) != "stale-should-go"
        for cb in litellm.success_callback + litellm._async_success_callback
    )


def test_success_callback_reload_is_idempotent_without_duplicates():
    proxy_config = ProxyConfig()
    first = _MarkerSuccessLogger(name="reload-me")
    second = _MarkerSuccessLogger(name="reload-me")

    litellm.logging_callback_manager.add_litellm_success_callback(first)
    litellm.logging_callback_manager.add_litellm_async_success_callback(first)
    assert _count_instances(litellm.success_callback, _MarkerSuccessLogger) == 1

    # Reload: clear event lists, re-add via manager (dedupe by custom-logger key).
    proxy_config._reset_event_callback_lists("success")
    litellm.logging_callback_manager.add_litellm_success_callback(second)
    litellm.logging_callback_manager.add_litellm_async_success_callback(second)
    # Calling add again must not create duplicates.
    litellm.logging_callback_manager.add_litellm_success_callback(second)
    litellm.logging_callback_manager.add_litellm_async_success_callback(second)

    assert _count_instances(litellm.success_callback, _MarkerSuccessLogger) == 1
    assert _count_instances(litellm._async_success_callback, _MarkerSuccessLogger) == 1


def test_success_callback_removal_clears_sync_and_async_lists_together():
    """Sync and async lists must not drift when a success_callback is removed."""
    proxy_config = ProxyConfig()
    keep = _MarkerSuccessLogger(name="keep")
    drop = _MarkerSuccessLogger(name="drop")

    for cb in (keep, drop):
        litellm.logging_callback_manager.add_litellm_success_callback(cb)
        litellm.logging_callback_manager.add_litellm_async_success_callback(cb)

    assert _count_instances(litellm.success_callback, _MarkerSuccessLogger) == 2
    assert _count_instances(litellm._async_success_callback, _MarkerSuccessLogger) == 2

    proxy_config._reset_event_callback_lists("success")
    litellm.logging_callback_manager.add_litellm_success_callback(keep)
    litellm.logging_callback_manager.add_litellm_async_success_callback(keep)

    assert _count_instances(litellm.success_callback, _MarkerSuccessLogger) == 1
    assert _count_instances(litellm._async_success_callback, _MarkerSuccessLogger) == 1
    assert all(
        getattr(cb, "name", None) != "drop"
        for cb in litellm.success_callback + litellm._async_success_callback
    )
    assert any(getattr(cb, "name", None) == "keep" for cb in litellm.success_callback)
    assert any(
        getattr(cb, "name", None) == "keep" for cb in litellm._async_success_callback
    )


def test_failure_callback_reset_is_symmetric_for_async_list():
    proxy_config = ProxyConfig()
    failure_logger = _MarkerSuccessLogger(name="failure-only")

    litellm.logging_callback_manager.add_litellm_failure_callback(failure_logger)
    litellm.logging_callback_manager.add_litellm_async_failure_callback(failure_logger)
    assert len(litellm.failure_callback) == 1
    assert len(litellm._async_failure_callback) == 1

    proxy_config._reset_event_callback_lists("failure")
    assert litellm.failure_callback == []
    assert litellm._async_failure_callback == []


def test_failure_callback_reset_preserves_general_callbacks_only():
    proxy_config = ProxyConfig()
    general = _MarkerGeneralLogger(name="general-failure")
    orphan = _MarkerSuccessLogger(name="orphan-failure")

    litellm.callbacks = [general]
    litellm.logging_callback_manager.add_litellm_failure_callback(general)
    litellm.logging_callback_manager.add_litellm_async_failure_callback(general)
    litellm.logging_callback_manager.add_litellm_failure_callback(orphan)
    litellm.logging_callback_manager.add_litellm_async_failure_callback(orphan)

    proxy_config._reset_event_callback_lists("failure")

    assert _count_instances(litellm.failure_callback, _MarkerGeneralLogger) == 1
    assert _count_instances(litellm._async_failure_callback, _MarkerGeneralLogger) == 1
    assert _count_instances(litellm.failure_callback, _MarkerSuccessLogger) == 0
    assert _count_instances(litellm._async_failure_callback, _MarkerSuccessLogger) == 0


@pytest.mark.asyncio
async def test_load_config_success_callback_branch_preserves_general_and_is_idempotent():
    """End-to-end load_config success_callback processing for combined surfaces."""
    proxy_config = ProxyConfig()
    general_logger = _MarkerGeneralLogger(name="general-via-callbacks-list")
    success_logger = _MarkerSuccessLogger(name="success-via-dotted")

    # Pre-seed general callbacks ownership the way the `callbacks` path would.
    litellm.callbacks = [general_logger]
    litellm.logging_callback_manager.add_litellm_success_callback(general_logger)
    litellm.logging_callback_manager.add_litellm_async_success_callback(general_logger)

    config_content = {
        "model_list": [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"},
            }
        ],
        "litellm_settings": {
            "success_callback": ["tests.fake.SuccessLogger"],
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        yaml.dump(config_content, temp_file)
        temp_file_path = temp_file.name

    try:
        with patch(
            "litellm.proxy.proxy_server.get_instance_fn",
            return_value=success_logger,
        ):
            await proxy_config.load_config(
                router=None, config_file_path=temp_file_path
            )
            # Second load must remain idempotent (no duplicate success loggers).
            await proxy_config.load_config(
                router=None, config_file_path=temp_file_path
            )
    finally:
        os.unlink(temp_file_path)

    assert _count_instances(litellm.success_callback, _MarkerGeneralLogger) == 1
    assert _count_instances(litellm._async_success_callback, _MarkerGeneralLogger) == 1
    assert _count_instances(litellm.success_callback, _MarkerSuccessLogger) == 1
    assert _count_instances(litellm._async_success_callback, _MarkerSuccessLogger) == 1
