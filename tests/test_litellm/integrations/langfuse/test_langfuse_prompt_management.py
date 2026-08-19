import asyncio
import datetime
import logging
import threading
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.integrations.langfuse import langfuse_prompt_management as langfuse_pm
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


class TestLangfusePromptManagement:
    def setup_method(self):
        # Mock langfuse package to avoid triggering real import.
        # The real langfuse import fails on Python 3.14 due to pydantic v1 incompatibility.
        # This also prevents test-ordering issues when earlier tests remove sys.modules["langfuse"].
        self._mock_langfuse = MagicMock()
        self._mock_langfuse.version.__version__ = "3.0.0"
        self._langfuse_patcher = patch.dict(
            "sys.modules", {"langfuse": self._mock_langfuse}
        )
        self._langfuse_patcher.start()

    def teardown_method(self):
        self._langfuse_patcher.stop()

    def test_get_prompt_from_id(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with patch.object(
            langfuse_prompt_management, "should_run_prompt_management"
        ) as mock_should_run_prompt_management, patch.object(
            langfuse_prompt_management, "_get_prompt_from_id"
        ) as mock_get_prompt_from_id:
            mock_should_run_prompt_management.return_value = True
            langfuse_prompt_management.get_chat_completion_prompt(
                model="langfuse/langfuse-model",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                non_default_params={},
                prompt_id="test-chat-prompt",
                prompt_variables={},
                dynamic_callback_params={},
                prompt_version=4,
            )

            mock_get_prompt_from_id.assert_called_once()
            assert mock_get_prompt_from_id.call_args.kwargs["prompt_version"] == 4

    def test_log_failure_event_runs_async_logger(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with patch.object(
            langfuse_prompt_management,
            "async_log_failure_event",
            new_callable=AsyncMock,
        ) as mock_async_log:
            kwargs = {"standard_callback_dynamic_params": {}}
            start_time, end_time = 1, 2

            langfuse_prompt_management.log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=start_time,
                end_time=end_time,
            )

            mock_async_log.assert_awaited_once()

    def test_async_log_success_event_defaults_missing_dynamic_params(self):
        langfuse_prompt_management = LangfusePromptManagement()
        mock_logger = MagicMock()

        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseHandler"
        ) as mock_handler:
            mock_handler.get_langfuse_logger_for_request.return_value = mock_logger

            asyncio.run(
                langfuse_prompt_management.async_log_success_event(
                    kwargs={"user": "test-user"},
                    response_obj={"ok": True},
                    start_time=datetime.datetime.now(datetime.UTC),
                    end_time=datetime.datetime.now(datetime.UTC),
                )
            )

        mock_handler.get_langfuse_logger_for_request.assert_called_once()
        assert (
            mock_handler.get_langfuse_logger_for_request.call_args.kwargs[
                "standard_callback_dynamic_params"
            ]
            == {}
        )
        mock_logger.log_event_on_langfuse.assert_called_once()

    def test_log_success_event_without_running_loop_does_not_error(self, caplog):
        langfuse_prompt_management = LangfusePromptManagement()
        mock_logger = MagicMock()
        previous_loop = GLOBAL_LOGGING_WORKER._bound_loop
        GLOBAL_LOGGING_WORKER._bound_loop = None

        try:
            with patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseHandler"
            ) as mock_handler, caplog.at_level(logging.ERROR, logger="LiteLLM"):
                mock_handler.get_langfuse_logger_for_request.return_value = mock_logger
                langfuse_prompt_management.log_success_event(
                    kwargs={"user": "test-user"},
                    response_obj={"ok": True},
                    start_time=datetime.datetime.now(datetime.UTC),
                    end_time=datetime.datetime.now(datetime.UTC),
                )
        finally:
            GLOBAL_LOGGING_WORKER._bound_loop = previous_loop

        mock_logger.log_event_on_langfuse.assert_called_once()
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "no running event loop" not in logged
        assert not any(record.exc_info for record in caplog.records)

    def test_langfuse_failure_without_running_loop_does_not_chain_event_loop_error(
        self, caplog
    ):
        langfuse_prompt_management = LangfusePromptManagement()
        mock_logger = MagicMock()
        mock_logger.log_event_on_langfuse.side_effect = RuntimeError(
            "dictionary changed size during iteration"
        )
        previous_loop = GLOBAL_LOGGING_WORKER._bound_loop
        GLOBAL_LOGGING_WORKER._bound_loop = None

        try:
            with patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseHandler"
            ) as mock_handler, caplog.at_level(logging.ERROR, logger="LiteLLM"):
                mock_handler.get_langfuse_logger_for_request.return_value = mock_logger
                langfuse_prompt_management.log_success_event(
                    kwargs={"user": "test-user"},
                    response_obj={"ok": True},
                    start_time=datetime.datetime.now(datetime.UTC),
                    end_time=datetime.datetime.now(datetime.UTC),
                )
        finally:
            GLOBAL_LOGGING_WORKER._bound_loop = previous_loop

        logged = "\n".join(
            record.getMessage()
            + (
                "".join(traceback.format_exception(*record.exc_info))
                if record.exc_info
                else ""
            )
            for record in caplog.records
        )
        assert "no running event loop" not in logged

    def test_run_langfuse_async_event_schedules_onto_proxy_loop(self):
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        previous_loop = GLOBAL_LOGGING_WORKER._bound_loop
        GLOBAL_LOGGING_WORKER._bound_loop = loop
        ran_on = {}

        async def marker():
            ran_on["loop_id"] = id(asyncio.get_running_loop())
            return "ok"

        try:
            result = langfuse_pm._run_langfuse_async_event(marker)
            assert result == "ok"
            assert ran_on["loop_id"] == id(loop)
        finally:
            GLOBAL_LOGGING_WORKER._bound_loop = previous_loop
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()
