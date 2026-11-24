import pytest

from litellm.integrations.aawm_session_history import writer
from litellm.integrations.aawm_session_history.waits import SessionHistoryWaitMiddleware


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [422, None])
async def test_terminal_wait_checkpoint_retains_sent_status(monkeypatch, status_code):
    checkpoints = []
    sent_messages = []
    monkeypatch.setattr(writer, "_build_session_history_dsn", lambda: "test-dsn")
    monkeypatch.setattr(writer, "_enqueue_session_history_record", checkpoints.append)

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    async def send(message):
        sent_messages.append(message)

    async def app(scope, receive, send):
        await receive()
        if status_code is None:
            raise RuntimeError("Failed before headers")
        await send({"type": "http.response.start", "status": status_code, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = SessionHistoryWaitMiddleware(app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [(b"x-litellm-call-id", b"test-call")],
    }
    if status_code is None:
        with pytest.raises(RuntimeError, match="Failed before headers"):
            await middleware(scope, receive, send)
        assert not sent_messages
    else:
        await middleware(scope, receive, send)
        assert sent_messages[0]["status"] == status_code

    checkpoint = checkpoints[-1]
    assert checkpoint["litellm_call_id"] == "test-call"
    wait = checkpoint["metadata"]["wait_accounting"]
    assert wait["status"] == "failed"
    if status_code is None:
        assert "response_status_code" not in wait
    else:
        assert wait["response_status_code"] == status_code
