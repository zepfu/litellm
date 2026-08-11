"""A6: full-payload capture control-file trust, header drop, atomic 0600 writes."""

from __future__ import annotations

import base64
import json
import os
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import litellm.integrations.aawm_passthrough_shape_capture as capture


def test_full_payload_headers_drop_authorization() -> None:
    headers = {
        "Authorization": "Bearer secret-token",
        "x-api-key": "sk-test",
        "content-type": "application/json",
        "x-request-id": "abc",
    }
    sanitized = capture._full_payload_headers(headers)
    lower_keys = {k.lower() for k in sanitized}
    assert "authorization" not in lower_keys
    assert "x-api-key" not in lower_keys
    assert "content-type" in lower_keys
    assert sanitized["content-type"] == "application/json"

    items = capture._full_payload_header_items(headers)
    names = {item["name"].lower() for item in items}
    assert "authorization" not in names
    assert "x-api-key" not in names


def test_untrusted_world_writable_control_file_is_ignored(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(capture._FULL_PAYLOAD_ENV_FLAG, raising=False)
    # Simulate default /tmp control: parent world-writable
    parent = tmp_path / "world"
    parent.mkdir()
    os.chmod(parent, 0o777)
    control = parent / "pass_through_full_payloads.enabled"
    control.write_text("1", encoding="utf-8")
    os.chmod(control, 0o644)

    monkeypatch.setenv(
        capture._FULL_PAYLOAD_CONTROL_FILE_ENV, str(control)
    )
    assert capture.passthrough_full_payload_capture_enabled() is False


def test_trusted_control_file_enables_capture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(capture._FULL_PAYLOAD_ENV_FLAG, raising=False)
    parent = tmp_path / "owned"
    parent.mkdir()
    os.chmod(parent, 0o700)
    control = parent / "enabled"
    control.write_text("1", encoding="utf-8")
    os.chmod(control, 0o600)

    monkeypatch.setenv(capture._FULL_PAYLOAD_CONTROL_FILE_ENV, str(control))
    assert capture.passthrough_full_payload_capture_enabled() is True


def test_env_flag_enables_without_control_file(monkeypatch) -> None:
    monkeypatch.setenv(capture._FULL_PAYLOAD_ENV_FLAG, "1")
    monkeypatch.delenv(capture._FULL_PAYLOAD_CONTROL_FILE_ENV, raising=False)
    assert capture.passthrough_full_payload_capture_enabled() is True


def test_write_full_payload_artifact_is_mode_0600(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(capture._FULL_PAYLOAD_ENV_FLAG, "1")
    monkeypatch.setenv(capture._FULL_PAYLOAD_DIR_ENV, str(tmp_path / "caps"))

    path_str = capture._write_full_payload_artifact(
        {
            "provider": "anthropic",
            "mode": "test",
            "litellm_call_id": "call-a6-1",
            "body": {"ok": True},
        }
    )
    assert path_str is not None
    path = Path(path_str)
    assert path.is_file()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["body"]["ok"] is True

    cap_dir = tmp_path / "caps"
    dir_mode = cap_dir.stat().st_mode & 0o777
    # owner rwx only preferred
    assert dir_mode & stat.S_IWOTH == 0
    assert dir_mode & stat.S_IXUSR


def test_full_payload_jsonable_utf8_limits_apply_to_nested_fields(monkeypatch) -> None:
    monkeypatch.setenv(capture._FULL_PAYLOAD_AGGREGATE_MAX_BYTES_ENV, "4096")
    monkeypatch.setenv(capture._FULL_PAYLOAD_FIELD_MAX_BYTES_ENV, "5")

    budget = capture._new_full_payload_budget()
    shaped = capture._jsonable_full_payload(
        {"root": "\U0001f680\U0001f680\U0001f680", "nest": {"msg": "ééééé"}},
        budget=budget,
        path="$",
        depth=0,
    )

    assert shaped["root"] == "\U0001f680"
    assert shaped["nest"]["msg"] == "éé"
    assert capture._truncate_to_byte_limit("éa", 1) == ""
    assert capture._truncate_to_byte_limit_with_counts("éa", 1) == ("", 0, 3)
    assert len(shaped["root"].encode("utf-8")) <= 5
    assert len(shaped["nest"]["msg"].encode("utf-8")) <= 5
    root_record = next(
        record for record in budget.truncations if record["path"] == "$.root"
    )
    assert root_record["original_bytes"] == 12
    assert root_record["stored_bytes"] == 4
    nest_record = next(
        record for record in budget.truncations if record["path"] == "$.nest.msg"
    )
    assert nest_record["original_bytes"] == 10
    assert nest_record["stored_bytes"] == 4

    record_paths = [record["path"] for record in budget.truncations]
    assert "$.root" in record_paths
    assert "$.nest.msg" in record_paths

    monkeypatch.setenv(capture._FULL_PAYLOAD_FIELD_MAX_BYTES_ENV, "49")
    multibyte_budget = capture._new_full_payload_budget(4096)
    multibyte = capture._jsonable_full_payload(
        {"payload": "😄" * 50},
        budget=multibyte_budget,
        path="$",
        depth=0,
    )
    assert multibyte["payload"] == "😄" * 12
    assert len(multibyte["payload"].encode("utf-8")) == 48
    assert multibyte_budget.truncations
    payload_record = next(
        record for record in multibyte_budget.truncations if record["path"] == "$.payload"
    )
    assert payload_record["original_bytes"] == 200
    assert payload_record["stored_bytes"] == 48


def test_full_payload_headers_bound_length_and_repetition_without_sensitive(monkeypatch) -> None:
    class _UnaccessedValue:
        def __str__(self) -> str:
            raise AssertionError("out-of-cap header value was accessed")

    class _LazyHeaders(Mapping[str, Any]):
        _names = (
            "Authorization",
            "x-api-key",
            "trace-parent",
            "x-repeat",
            "x-large",
            "x-over-cap-1",
            "x-over-cap-2",
        )
        _values = {
            "trace-parent": "A" * 32,
            "x-repeat": "B" * 16,
            "x-large": "\U0001f4a5" * 32,
        }

        def __getitem__(self, key: str) -> Any:
            if key in {"Authorization", "x-api-key"}:
                raise AssertionError("sensitive header value was accessed")
            if key.startswith("x-over-cap"):
                raise AssertionError("out-of-cap mapping value was accessed")
            return self._values[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._names)

        def __len__(self) -> int:
            return len(self._names)

        def items(self):
            raise AssertionError("headers.items() must not be materialized")

        def multi_items(self) -> Iterator[tuple[str, Any]]:
            yield "Authorization", _UnaccessedValue()
            yield "x-api-key", _UnaccessedValue()
            yield "trace-parent", "A" * 32
            yield "x-repeat", "B" * 16
            yield "x-repeat", "C" * 16
            yield "x-large", "\U0001f4a5" * 32
            yield "x-over-cap-1", _UnaccessedValue()
            yield "x-over-cap-2", _UnaccessedValue()

    headers = _LazyHeaders()

    monkeypatch.setattr(capture, "_FULL_PAYLOAD_MAX_HEADERS", 3)
    monkeypatch.setenv(capture._FULL_PAYLOAD_FIELD_MAX_BYTES_ENV, "12")

    sanitized = capture._full_payload_headers(headers)
    sanitized_keys = {k.lower() for k in sanitized if not k.startswith("_")}
    assert "authorization" not in sanitized_keys
    assert "x-api-key" not in sanitized_keys
    assert sanitized["_truncated_header_count"] == 2
    assert any(
        entry["reason"] == "header_value_bytes"
        for entry in sanitized.get("_truncated_headers", [])
    )
    large_header_entry = next(
        entry
        for entry in sanitized["_truncated_headers"]
        if entry["path"] == "$.headers.x-large"
    )
    assert large_header_entry["original_bytes"] == 128
    assert large_header_entry["stored_bytes"] == 12

    items = capture._full_payload_header_items(headers)
    assert items[-1] == {
        "path": "$.header_items",
        "reason": "header_count",
        "dropped_count": 3,
    }
    repeated = [item for item in items[:-1] if item["name"] == "x-repeat"]
    assert len(repeated) == 2
    assert any(item["truncated"] for item in items[:-1] if "truncated" in item)
    serialized = json.dumps({"headers": sanitized, "items": items})
    assert "should-never-appear" not in serialized
    assert "top-secret" not in serialized


def test_full_payload_depth_and_container_cardinality_limits(monkeypatch) -> None:
    class _LazyList(list):
        def __iter__(self):
            raise AssertionError("list values must not be eagerly iterated")

        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("list slices must not be materialized")
            if index >= 2:
                raise AssertionError("out-of-cap list value was accessed")
            return super().__getitem__(index)

    class _LazyMapping(Mapping[str, Any]):
        _names = ("items", "k1", "k2", "k3")

        def __init__(self) -> None:
            self._values = {
                "items": _LazyList([1, 2, 3, 4]),
                "k1": "v1",
                "k2": "v2",
            }

        def __getitem__(self, key: str) -> Any:
            if key == "k3":
                raise AssertionError("out-of-cap mapping value was accessed")
            return self._values[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._names)

        def __len__(self) -> int:
            return len(self._names)

        def keys(self):
            raise AssertionError("mapping keys must not be materialized")

    monkeypatch.setattr(capture, "_FULL_PAYLOAD_MAX_DEPTH", 3)
    monkeypatch.setattr(capture, "_FULL_PAYLOAD_MAX_DICT_KEYS", 3)
    monkeypatch.setattr(capture, "_FULL_PAYLOAD_MAX_LIST_ITEMS", 2)

    budget = capture._new_full_payload_budget(8192)
    shaped = capture._jsonable_full_payload(
        {
            "container": _LazyMapping(),
            "depth": {
                "level1": {"level2": {"level3": {"marker": "truncated"}}}
            },
        },
        budget=budget,
        path="$",
        depth=0,
    )

    assert shaped["container"]["items"] == [1, 2]
    assert shaped["depth"]["level1"]["level2"] == {
        "_truncated": True,
        "reason": "depth_limit",
    }
    assert "k3" not in shaped["container"]

    reasons = {record["reason"] for record in budget.truncations}
    assert "depth_limit" in reasons
    assert "list_item_limit" in reasons
    assert "dict_key_limit" in reasons

    access = {"getitem": 0}

    class _NestedEmptyList(list):
        def __init__(self, *, depth_remaining: int):
            self._child = [] if depth_remaining <= 1 else _NestedEmptyList(
                depth_remaining=depth_remaining - 1
            )

        def __getitem__(self, index: int) -> Any:
            if index >= 500:
                raise IndexError
            access["getitem"] += 1
            return self._child

        def __len__(self) -> int:
            return 500

    monkeypatch.setattr(capture, "_FULL_PAYLOAD_MAX_LIST_ITEMS", 500)
    monkeypatch.setattr(capture, "_FULL_PAYLOAD_MAX_DEPTH", 12)
    bounded_payload = capture._jsonable_full_payload(
        {"nested": _NestedEmptyList(depth_remaining=2)},
        budget=capture._new_full_payload_budget(3_000),
        path="$",
        depth=0,
    )

    assert bounded_payload is not capture._FULL_PAYLOAD_OMITTED
    assert isinstance(bounded_payload, Mapping)
    assert bounded_payload["nested"]
    assert isinstance(bounded_payload["nested"], list)
    assert len(bounded_payload["nested"]) <= 3
    assert access["getitem"] < 500

    byte_payload = capture._jsonable_full_payload(
        {"bytes": [b"" for _ in range(500)]},
        budget=capture._new_full_payload_budget(3000),
        path="$",
        depth=0,
    )
    assert byte_payload is not capture._FULL_PAYLOAD_OMITTED
    assert isinstance(byte_payload, Mapping)
    assert byte_payload["bytes"]
    assert len(byte_payload["bytes"]) < 500
    assert all(
        item["encoding"] == "base64" and item["data"] == "" and not item["truncated"]
        for item in byte_payload["bytes"]
    )
    byte_payload_bytes = len(
        json.dumps(byte_payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    assert byte_payload_bytes <= 3000


def test_full_payload_aggregate_ceiling_preserves_path_only_truncations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(capture._FULL_PAYLOAD_ENV_FLAG, "1")
    monkeypatch.setenv(capture._FULL_PAYLOAD_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(capture._ENV_FLAG, raising=False)
    monkeypatch.delenv(capture._DIAGNOSTIC_ENV_FLAG, raising=False)
    monkeypatch.setenv(capture._FULL_PAYLOAD_FIELD_MAX_BYTES_ENV, "8192")
    monkeypatch.setenv(
        capture._FULL_PAYLOAD_AGGREGATE_MAX_BYTES_ENV,
        "1",
    )

    minimum_path = capture.capture_passthrough_stream_shape(
        provider="test",
        request_body={"non_finite": float("nan")},
        all_chunks=["é" * 400],
        raw_bytes=[b"x" * 10_000],
        litellm_call_id="call-a6-min",
    )
    assert minimum_path is not None
    minimum_bytes = Path(minimum_path).read_bytes()
    minimum_payload = json.loads(minimum_bytes)
    assert capture._full_payload_aggregate_max_bytes() == 256
    assert len(minimum_bytes) <= 256
    assert minimum_payload["aggregate_limit_bytes"] == 256

    monkeypatch.setenv(
        capture._FULL_PAYLOAD_AGGREGATE_MAX_BYTES_ENV,
        "4096",
    )
    path_str = capture.capture_passthrough_stream_shape(
        provider="test",
        request_body={"non_finite": float("nan")},
        all_chunks=["é" * 400],
        raw_bytes=[b"x" * 10_000],
        litellm_call_id="call-a6-budget",
    )
    assert path_str is not None
    artifact_bytes = Path(path_str).read_bytes()
    artifact = json.loads(artifact_bytes)
    assert len(artifact_bytes) <= 4096
    assert capture._field_byte_limit() == 4096
    assert artifact["request"]["body"]["non_finite"] == "nan"
    stream = artifact["response"]["stream"]
    assert len(stream["lines"][0].encode("utf-8")) == 800
    stored_raw = base64.b64decode(stream["raw_chunks_base64"][0])
    assert 0 < len(stored_raw) < 10_000
    assert artifact["aggregate_truncated"] is True
    allowed_record_keys = {
        "path",
        "reason",
        "original_bytes",
        "stored_bytes",
        "dropped_count",
    }
    assert all(
        set(record) <= allowed_record_keys for record in artifact["truncations"]
    )
    assert all("note" not in record for record in artifact["truncations"])
