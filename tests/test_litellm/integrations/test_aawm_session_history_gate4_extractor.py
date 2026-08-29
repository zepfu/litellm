from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import stat
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm.integrations import (
    aawm_agent_identity as identity_host,
    aawm_session_history as session_history_package,
)
from litellm.integrations.aawm_session_history import runtime, spool


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "extract_aawm_session_history_gate4.py"
STATE_NAMES = (
    "_aawm_session_history_queue",
    "_aawm_session_history_worker_inflight_records",
    "_aawm_session_history_shutdown_lock",
    "_aawm_session_history_shutdown_in_progress",
)
WORKER_STATE_NAME = "_aawm_session_history_worker"
EXTRACTOR_SHA256 = hashlib.sha256(b"gate4-test-extractor").hexdigest()
ARTIFACT_NAMES = (
    "queued.jsonl",
    "inflight.jsonl",
    "manifest.json",
    "sha256sums.txt",
)


def load_extractor_source():
    raw_source = SCRIPT_PATH.read_text(encoding="utf-8")
    transport = base64.b64encode(zlib.compress(raw_source.encode("utf-8")))
    transport_literal = transport.decode("ascii")
    source = f"""
import base64
import zlib
_GATE4_SOURCE = zlib.decompress(base64.b64decode({transport_literal!r})).decode("utf-8")
exec(compile(_GATE4_SOURCE, "<aawm-session-history-gate4>", "exec"), globals())
del _GATE4_SOURCE
"""
    namespace: dict = {"__name__": "aawm_session_history_gate4_source"}
    exec(compile(source, "<aawm-session-history-gate4-wrapper>", "exec"), namespace)
    return namespace["extract_gate4"], raw_source


def validate_extractor_arguments(output_dir: Path) -> str:
    return os.path.abspath(output_dir)


def expect_ambiguous_state(extract_gate4, output_dir, name):
    with pytest.raises(RuntimeError, match=f"ambiguous session_history state: {name}"):
        extract_gate4(
            output_dir,
            lock_timeout_seconds=0.01,
            extractor_sha256=EXTRACTOR_SHA256,
        )


def set_runtime_state_without_mirroring(name, value):
    setattr(runtime, name, value)


@pytest.fixture
def live_state():
    previous = {name: getattr(runtime, name) for name in (*STATE_NAMES, WORKER_STATE_NAME)}
    worker_stop = threading.Event()
    worker = threading.Thread(
        target=worker_stop.wait,
        name="aawm-session-history-test-worker",
        daemon=True,
    )
    worker.start()
    try:
        runtime._set_state("_aawm_session_history_queue", queue.Queue(maxsize=8))
        runtime._set_state("_aawm_session_history_worker_inflight_records", [])
        runtime._set_state("_aawm_session_history_shutdown_in_progress", False)
        runtime._set_state("_aawm_session_history_worker", worker)
        yield worker_stop
    finally:
        worker_stop.set()
        worker.join(timeout=2)
        for name, value in previous.items():
            runtime._set_state(name, value)


def populate_records(queue_obj, inflight):
    queued = [
        {
            "litellm_call_id": "call-queued-1",
            "id": "record-queued-1",
            "session_id": "session-queued-1",
            "created_at": datetime(2026, 8, 28, 12, 0, 1, tzinfo=timezone.utc),
        },
        {
            "session_id": "shared-session",
        },
        {
            "litellm_call_id": "shared-call",
            "session_id": "shared-session",
        },
        {},
        None,
    ]
    inflight_records = [
        {
            "litellm_call_id": "call-inflight-1",
            "id": "record-inflight-1",
            "session_id": "session-inflight-1",
            "created_at": datetime(2026, 8, 28, 12, 0, 2, tzinfo=timezone.utc),
        },
        {
            "litellm_call_id": "shared-call",
            "session_id": "shared-session",
        },
    ]
    for record in queued:
        queue_obj.put(record)
    inflight.extend(inflight_records)
    return queued, inflight_records


def read_jsonl(path):
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        documents.append(json.loads(line))
    return documents


def read_records(path):
    documents = read_jsonl(path)
    assert documents[0]["type"] == "metadata"
    assert documents[0]["format_version"] == 1
    return [spool._decode_session_history_spool_value(document["record"]) for document in documents[1:]]


def assert_private_output(output_dir, extracted, manifest):
    directory_mode = stat.S_IMODE(output_dir.stat().st_mode)
    assert directory_mode == 0o700
    for name in ARTIFACT_NAMES:
        artifact_mode = stat.S_IMODE((output_dir / name).stat().st_mode)
        assert artifact_mode == 0o600
        assert not (output_dir / name).is_symlink()
    assert extracted["artifacts"]["manifest"] == str(output_dir / "manifest.json")
    assert manifest["disposition"] == "extracted_unvalidated"
    assert manifest["hashes"]["extractor_sha256"] == EXTRACTOR_SHA256
    assert (
        manifest["hashes"]["format_sha256"] == hashlib.sha256(b"aawm-session-history-gate4-spool-format-v1").hexdigest()
    )

    expected_checksums = ""
    for name in ARTIFACT_NAMES[:2]:
        payload = (output_dir / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected_checksums += f"{digest}  {name}\n"
        assert manifest["artifacts"][name] == {
            "sha256": digest,
            "size_bytes": len(payload),
        }
    manifest_payload = (output_dir / "manifest.json").read_bytes()
    manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
    expected_checksums += f"{manifest_digest}  manifest.json\n"
    assert (output_dir / "sha256sums.txt").read_text(encoding="utf-8") == (expected_checksums)


def test_gate4_extractor_source_injection_preserves_and_exports_records(tmp_path, live_state):
    extract_gate4, raw_source = load_extractor_source()
    assert "import litellm" not in raw_source
    assert "from litellm" not in raw_source

    queue_obj = runtime._aawm_session_history_queue
    inflight = runtime._aawm_session_history_worker_inflight_records
    queued, inflight_records = populate_records(queue_obj, inflight)
    original_queued = list(queue_obj.queue)
    original_unfinished = queue_obj.unfinished_tasks
    output_dir = tmp_path / "gate4-success"

    extracted = extract_gate4(
        str(output_dir),
        lock_timeout_seconds=1.0,
        extractor_sha256=EXTRACTOR_SHA256,
    )

    assert queue_obj.qsize() == 5
    assert list(queue_obj.queue) == original_queued
    assert queue_obj.unfinished_tasks == original_unfinished
    assert inflight is runtime._aawm_session_history_worker_inflight_records
    assert inflight == inflight_records
    assert all(record is queued[index] for index, record in enumerate(original_queued[:3]))
    assert runtime._aawm_session_history_shutdown_in_progress is True

    decoded_queued = read_records(output_dir / "queued.jsonl")
    decoded_inflight = read_records(output_dir / "inflight.jsonl")
    assert decoded_queued == queued[:4]
    assert decoded_inflight == inflight_records
    assert decoded_queued[0]["created_at"] == queued[0]["created_at"]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert_private_output(output_dir, extracted, manifest)
    assert manifest["counts"] == {
        "queued": 4,
        "inflight": 2,
        "sentinels_excluded": 1,
        "identities": 11,
        "missing_identities": 1,
    }
    assert manifest["queue"] == {
        "size_before_copy": 5,
        "size_after_copy": 5,
        "maxsize": 8,
        "unfinished_tasks": original_unfinished,
    }
    assert manifest["worker"]["present"] is True
    assert manifest["worker"]["name"] == "aawm-session-history-test-worker"
    assert manifest["worker"]["ident"] == runtime._aawm_session_history_worker.ident
    assert manifest["identities"]["missing"] == [{"set": "queued", "index": 3}]
    assert manifest["identities"]["duplicates"]["within"] == [
        {
            "field": "session_id",
            "value": "shared-session",
            "occurrences": {
                "queued": [1, 2],
                "inflight": [1],
            },
        }
    ]
    assert manifest["identities"]["duplicates"]["across"] == [
        {
            "field": "session_id",
            "value": "shared-session",
            "occurrences": {"queued": [1, 2], "inflight": [1]},
        },
        {
            "field": "litellm_call_id",
            "value": "shared-call",
            "occurrences": {"queued": [2], "inflight": [1]},
        },
    ]


def test_gate4_extractor_accepts_stale_identity_worker_metadata_binding(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    output_dir = tmp_path / "stale-identity-worker"
    monkeypatch.setattr(identity_host, WORKER_STATE_NAME, None)

    extracted = extract_gate4(
        str(output_dir),
        lock_timeout_seconds=1.0,
        extractor_sha256=EXTRACTOR_SHA256,
    )

    assert extracted["artifacts"]["manifest"] == str(output_dir / "manifest.json")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["worker"]["present"] is True
    assert manifest["worker"]["name"] == runtime._aawm_session_history_worker.name
    assert manifest["worker"]["ident"] == runtime._aawm_session_history_worker.ident


def test_gate4_extractor_fails_closed_on_shutdown_lock_timeout(tmp_path, live_state):
    extract_gate4, _ = load_extractor_source()
    output_dir = tmp_path / "shutdown-timeout"
    lock = runtime._aawm_session_history_shutdown_lock

    with lock:
        with pytest.raises(TimeoutError, match="shutdown lock"):
            extract_gate4(
                str(output_dir),
                lock_timeout_seconds=0.01,
                extractor_sha256=EXTRACTOR_SHA256,
            )

    assert not output_dir.exists()
    assert runtime._aawm_session_history_shutdown_in_progress is False


def test_gate4_extractor_captures_batch_replaced_before_shutdown_lock_acquires(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    output_dir = tmp_path / "post-lock-replacement"
    original_queue = runtime._aawm_session_history_queue
    original_inflight = runtime._aawm_session_history_worker_inflight_records
    original_queue.put({"litellm_call_id": "pre-lock-record"})
    original_inflight.append({"litellm_call_id": "pre-lock-inflight"})

    replacement_queue = queue.Queue(maxsize=4)
    replacement_inflight = []
    replacement_stop = threading.Event()
    replacement_worker = threading.Thread(
        target=replacement_stop.wait,
        name="aawm-session-history-test-worker-replacement",
        daemon=True,
    )
    replacement_worker.start()

    replaced_during_shutdown_lock = threading.Event()
    shutdown_lock_release = threading.Event()
    original_lock_acquire = runtime._aawm_session_history_shutdown_lock.acquire

    class ReplacementShutdownLock:
        def acquire(self, blocking=True, timeout=None):
            acquired = original_lock_acquire(blocking=blocking, timeout=timeout)
            if acquired:
                runtime._set_state("_aawm_session_history_queue", replacement_queue)
                runtime._set_state("_aawm_session_history_worker_inflight_records", replacement_inflight)
                runtime._set_state("_aawm_session_history_worker", replacement_worker)
                replaced_during_shutdown_lock.set()
                shutdown_lock_release.wait(timeout=1)
            return acquired

        def release(self):
            shutdown_lock_release.set()
            original_lock_release()

    original_lock_release = runtime._aawm_session_history_shutdown_lock.release
    replacement_shutdown_lock = ReplacementShutdownLock()
    monkeypatch.setattr(runtime, "_aawm_session_history_shutdown_lock", replacement_shutdown_lock)
    runtime._set_state("_aawm_session_history_shutdown_lock", replacement_shutdown_lock)
    monkeypatch.setattr(identity_host, "_aawm_session_history_shutdown_lock", replacement_shutdown_lock)

    extracted = extract_gate4(
        str(output_dir),
        lock_timeout_seconds=1.0,
        extractor_sha256=EXTRACTOR_SHA256,
    )

    assert replaced_during_shutdown_lock.is_set()

    assert extracted["queued_count"] == 0
    assert extracted["inflight_count"] == 0
    assert list(original_queue.queue) == [{"litellm_call_id": "pre-lock-record"}]
    assert original_inflight == [{"litellm_call_id": "pre-lock-inflight"}]
    assert not replacement_inflight
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["worker"]["name"] == ("aawm-session-history-test-worker-replacement")
    assert read_records(output_dir / "queued.jsonl") == []
    assert read_records(output_dir / "inflight.jsonl") == []


def test_gate4_extractor_fails_closed_on_queue_mutex_timeout(tmp_path, live_state):
    extract_gate4, _ = load_extractor_source()
    output_dir = tmp_path / "queue-timeout"
    queue_obj = runtime._aawm_session_history_queue

    with queue_obj.mutex:
        with pytest.raises(TimeoutError, match="queue mutex"):
            extract_gate4(
                str(output_dir),
                lock_timeout_seconds=0.01,
                extractor_sha256=EXTRACTOR_SHA256,
            )

    assert runtime._aawm_session_history_shutdown_in_progress is True
    assert not output_dir.exists()


def test_gate4_extractor_writes_atomic_events_in_order_and_rejects_preexisting_output(
    tmp_path, live_state, monkeypatch
):
    extract_gate4, _ = load_extractor_source()
    runtime._set_state("_aawm_session_history_worker_inflight_records", [])
    queue_obj = runtime._aawm_session_history_queue
    queue_obj.put({"litellm_call_id": "call-one"})

    persistence_events = []
    original_fsync = os.fsync
    original_rename = os.rename

    def descriptor_name(descriptor):
        return Path(os.readlink(f"/proc/self/fd/{descriptor}")).name

    def record_file_fsync(descriptor):
        persistence_events.append(("file_fsync", descriptor_name(descriptor)))
        return original_fsync(descriptor)

    def record_directory_fsync(descriptor):
        persistence_events.append(("directory_fsync", descriptor_name(descriptor)))
        return original_fsync(descriptor)

    def dispatch_fsync(descriptor):
        if descriptor_name(descriptor).startswith("."):
            return record_file_fsync(descriptor)
        return record_directory_fsync(descriptor)

    def normalize_event(event):
        if event[0] == "directory_fsync":
            return ("directory_fsync", output_dir.name)
        for artifact_name in ARTIFACT_NAMES:
            temp_name = f".{artifact_name}."
            if event[1].startswith(temp_name):
                if event[0] == "rename":
                    return ("rename", temp_name, event[2])
                return (event[0], temp_name)
        raise AssertionError(f"unexpected atomic persistence event: {event}")

    def normalize_persistence_events(events):
        return [normalize_event(event) for event in events]

    def record_rename(source, destination):
        persistence_events.append(("rename", os.path.basename(source), os.path.basename(destination)))
        return original_rename(source, destination)

    monkeypatch.setattr(os, "fsync", dispatch_fsync)
    monkeypatch.setattr(os, "rename", record_rename)
    output_dir = tmp_path / "gate4-write"
    extract_gate4(
        str(output_dir),
        lock_timeout_seconds=1.0,
        extractor_sha256=EXTRACTOR_SHA256,
    )

    expected_events = []
    for artifact_name in ARTIFACT_NAMES:
        temp_name = f".{artifact_name}."
        expected_events.extend(
            [
                ("file_fsync", temp_name),
                ("rename", temp_name, artifact_name),
                ("directory_fsync", output_dir.name),
            ]
        )
    assert normalize_persistence_events(persistence_events) == expected_events
    assert all(event[0] in {"file_fsync", "rename", "directory_fsync"} for event in persistence_events)

    before = {name: (output_dir / name).read_bytes() for name in ARTIFACT_NAMES}
    with pytest.raises(FileExistsError):
        extract_gate4(
            str(output_dir),
            lock_timeout_seconds=1.0,
            extractor_sha256=EXTRACTOR_SHA256,
        )
    after = {name: (output_dir / name).read_bytes() for name in ARTIFACT_NAMES}
    assert before == after


def test_gate4_extractor_fails_closed_on_file_fsync_failure(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    runtime._set_state("_aawm_session_history_worker_inflight_records", [])
    queue_obj = runtime._aawm_session_history_queue
    queue_obj.put({"litellm_call_id": "file-fsync-failure"})
    output_dir = tmp_path / "file-fsync-failure"

    def fail_file_fsync(descriptor):
        raise OSError("file fsync failed")

    monkeypatch.setattr(os, "fsync", fail_file_fsync)
    with pytest.raises(OSError, match="file fsync failed"):
        extract_gate4(
            str(output_dir),
            lock_timeout_seconds=1.0,
            extractor_sha256=EXTRACTOR_SHA256,
        )

    assert list(output_dir.iterdir()) == []
    for artifact_name in ARTIFACT_NAMES:
        assert not (output_dir / artifact_name).exists()
    assert runtime._aawm_session_history_shutdown_in_progress is True


def test_gate4_extractor_fails_closed_on_directory_fsync_failure(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    runtime._set_state("_aawm_session_history_worker_inflight_records", [])
    queue_obj = runtime._aawm_session_history_queue
    queue_obj.put({"litellm_call_id": "directory-fsync-failure"})
    output_dir = tmp_path / "directory-fsync-failure"
    renames = []
    original_rename = os.rename
    original_fsync = os.fsync

    def descriptor_name(descriptor):
        return Path(os.readlink(f"/proc/self/fd/{descriptor}")).name

    def fail_only_directory_fsync(descriptor):
        if descriptor_name(descriptor) == output_dir.name:
            raise OSError("directory fsync failed")
        return original_fsync(descriptor)

    def record_first_rename(source, destination):
        renames.append(destination)
        original_rename(source, destination)

    monkeypatch.setattr(os, "fsync", fail_only_directory_fsync)
    monkeypatch.setattr(os, "rename", record_first_rename)
    with pytest.raises(OSError, match="directory fsync failed"):
        extract_gate4(
            str(output_dir),
            lock_timeout_seconds=1.0,
            extractor_sha256=EXTRACTOR_SHA256,
        )

    assert renames == [str(output_dir / "queued.jsonl")]
    assert (output_dir / "queued.jsonl").is_file()
    for artifact_name in ARTIFACT_NAMES[1:]:
        assert not (output_dir / artifact_name).exists()
    assert runtime._aawm_session_history_shutdown_in_progress is True


def test_gate4_extractor_fails_closed_on_rename_failure(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    runtime._set_state("_aawm_session_history_worker_inflight_records", [])
    queue_obj = runtime._aawm_session_history_queue
    queue_obj.put({"litellm_call_id": "rename-failure"})
    output_dir = tmp_path / "rename-failure"
    renamed = []
    original_rename = os.rename

    def fail_second_rename(source, destination):
        if renamed:
            raise OSError("rename failed")
        renamed.append(destination)
        original_rename(source, destination)

    monkeypatch.setattr(os, "rename", fail_second_rename)
    with pytest.raises(OSError, match="rename failed"):
        extract_gate4(
            str(output_dir),
            lock_timeout_seconds=1.0,
            extractor_sha256=EXTRACTOR_SHA256,
        )

    assert renamed == [str(output_dir / "queued.jsonl")]
    assert (output_dir / "queued.jsonl").is_file()
    for artifact_name in ARTIFACT_NAMES[1:]:
        assert not (output_dir / artifact_name).exists()
    assert runtime._aawm_session_history_shutdown_in_progress is True


def test_gate4_extractor_validates_extractor_hash(tmp_path, live_state):
    extract_gate4, _ = load_extractor_source()
    output_dir = tmp_path / "invalid-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        extract_gate4(
            str(output_dir),
            lock_timeout_seconds=1.0,
            extractor_sha256="not-a-sha256",
        )
    assert not output_dir.exists()
    assert identity_host._aawm_session_history_queue is (runtime._aawm_session_history_queue)


def test_gate4_extractor_rejects_symlink_ancestor_before_lock(tmp_path, live_state):
    extract_gate4, _ = load_extractor_source()
    real_directory = tmp_path / "real-output-parent"
    real_directory.mkdir(mode=0o700)
    link = tmp_path / "link-to-parent"
    link.symlink_to(real_directory, target_is_directory=True)
    output_dir = link / "gate4-output"
    shutdown_lock = runtime._aawm_session_history_shutdown_lock
    queue_obj = runtime._aawm_session_history_queue

    with shutdown_lock, queue_obj.mutex:
        with pytest.raises(RuntimeError, match="symlink component"):
            extract_gate4(
                str(output_dir),
                lock_timeout_seconds=0.01,
                extractor_sha256=EXTRACTOR_SHA256,
            )

    assert not output_dir.exists()
    assert not real_directory.joinpath("gate4-output").exists()
    assert runtime._aawm_session_history_shutdown_in_progress is False


def test_gate4_extractor_fails_closed_on_mismatched_authoritative_identity_state(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    output_dir = tmp_path / "mismatched-identity-state"
    validated_output = validate_extractor_arguments(output_dir)

    for name in STATE_NAMES:
        if name == "_aawm_session_history_shutdown_in_progress":
            # Runtime's real _set_state mirrors to identity before the extractor
            # can observe a stale boolean, so emulate only the missing mirror.
            monkeypatch.setattr(runtime, "_set_state", set_runtime_state_without_mirroring)
        monkeypatch.setattr(identity_host, name, object())
        expect_ambiguous_state(extract_gate4, validated_output, name)
        monkeypatch.undo()
        assert not output_dir.exists()

    identity_host._set_state = lambda *args, **kwargs: None
    try:
        expect_ambiguous_state(extract_gate4, validated_output, "_set_state")
    finally:
        del identity_host._set_state
    assert not output_dir.exists()


def test_gate4_extractor_rejects_active_spool_containment(tmp_path, live_state, monkeypatch):
    extract_gate4, _ = load_extractor_source()
    active_spool_dir = tmp_path / "active-spool"
    active_spool_dir.mkdir(mode=0o700)
    output_dir = active_spool_dir / "gate4-output"
    shutdown_lock = runtime._aawm_session_history_shutdown_lock
    queue_obj = runtime._aawm_session_history_queue
    shared_resolver = lambda: str(active_spool_dir)  # noqa: E731
    monkeypatch.setattr(spool, "_get_session_history_spool_dir", shared_resolver)
    monkeypatch.setattr(session_history_package, "_get_session_history_spool_dir", shared_resolver)
    monkeypatch.setattr(identity_host, "_get_session_history_spool_dir", shared_resolver)

    with shutdown_lock, queue_obj.mutex:
        with pytest.raises(RuntimeError, match="inside active session_history spool"):
            extract_gate4(
                str(output_dir),
                lock_timeout_seconds=0.01,
                extractor_sha256=EXTRACTOR_SHA256,
            )

    assert not output_dir.exists()
    assert runtime._aawm_session_history_shutdown_in_progress is False
