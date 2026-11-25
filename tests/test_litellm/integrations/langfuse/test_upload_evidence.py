import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from packaging.version import Version

import litellm
from litellm.integrations.langfuse import langfuse as integration


def test_sdk_ingestion_failure_and_recovery_share_batch_identity(caplog, monkeypatch):
    version = pytest.importorskip("langfuse.version")
    if Version(version.__version__).major != 2:
        pytest.skip("requires Langfuse v2 ingestion")
    sent_batches = []

    def send(request):
        if request.url.path != "/api/public/ingestion":
            return httpx.Response(200, json={"data": [{"id": "probe", "name": "Probe"}]})
        sent_batches.append([event["id"] for event in json.loads(request.content)["batch"]])
        if len(sent_batches) == 1:
            return httpx.Response(503, json={"message": "private-response-marker"})
        return httpx.Response(201, json={})

    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    monkeypatch.setattr(integration, "should_use_langfuse_mock", lambda: False)
    monkeypatch.delenv("UPSTREAM_LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "prior-project")
    monkeypatch.delenv("LANGFUSE_DEBUG", raising=False)
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "1")
    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        monkeypatch.setattr(integration, "_get_httpx_client", lambda: SimpleNamespace(client=client))
        logger = integration.LangFuseLogger(
            langfuse_public_key="pk-lf-probe",
            langfuse_secret="sk-lf-private-marker",
            langfuse_host="https://langfuse.invalid",
        )
        try:
            with caplog.at_level("WARNING", logger="langfuse"):
                logger.Langfuse.trace(id=str(uuid4()), input="private-input-marker")
                logger.Langfuse.flush()
        finally:
            logger.Langfuse.shutdown()

    assert len(sent_batches) == 2
    assert sent_batches[0] == sent_batches[1]
    failure, recovery = [record for record in caplog.records if hasattr(record, "langfuse_upload_batch_id")]
    assert failure.langfuse_upload_outcome == "http_rejected"
    assert failure.langfuse_upload_statuses == [503]
    assert recovery.langfuse_upload_outcome == "accepted_after_failure"
    assert recovery.langfuse_upload_statuses == [201]
    assert failure.langfuse_upload_batch_id == recovery.langfuse_upload_batch_id
    assert failure.langfuse_upload_event_ids == sent_batches[0]
    assert failure.langfuse_upload_batch_size == len(sent_batches[0])
    assert "private-" not in repr([vars(failure), vars(recovery)])


def test_sdk_terminal_failure_before_http_retains_batch_evidence(caplog, monkeypatch):
    version = pytest.importorskip("langfuse.version")
    if Version(version.__version__).major != 2:
        pytest.skip("requires Langfuse v2 ingestion")
    from langfuse._task_manager import ingestion_consumer

    paths = []

    def send(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "probe", "name": "Probe"}]})

    def fail_before_post(**_):
        raise RuntimeError("private-pre-upload-marker")

    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    monkeypatch.setattr(integration, "should_use_langfuse_mock", lambda: False)
    monkeypatch.delenv("UPSTREAM_LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(ingestion_consumer, "IngestionMetadata", fail_before_post)
    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        monkeypatch.setattr(integration, "_get_httpx_client", lambda: SimpleNamespace(client=client))
        logger = integration.LangFuseLogger(
            langfuse_public_key="pk-lf-probe",
            langfuse_secret="sk-lf-private-marker",
            langfuse_host="https://langfuse.invalid",
        )
        try:
            logger.Langfuse.trace(id=str(uuid4()), input="private-input-marker")
            logger.Langfuse.flush()
        finally:
            logger.Langfuse.shutdown()

    assert "/api/public/ingestion" not in paths
    record, = [
        row for row in caplog.records
        if getattr(row, "langfuse_upload_outcome", None) == "terminal_sdk_error"
    ]
    assert record.langfuse_upload_batch_size == len(record.langfuse_upload_event_ids) == 1
    assert len(record.langfuse_upload_batch_id) == 64
    assert record.langfuse_upload_exception_type == "RuntimeError"
    assert record.langfuse_upload_post_entered is False
    assert "private-" not in repr(vars(record))
