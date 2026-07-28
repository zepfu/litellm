"""Focused tests for NVIDIA NIM embedding provider-policy layer (D1-542 body A).

Covers the bounded value-free AdaptationCollector, the atomic strict/drop
mapper, explicit extra_body policy, and a single D1-541 compatibility smoke.
D1-541 state isolation is otherwise covered by tests/llm_translation/test_nvidia_nim.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.param_adaptation import (
    MAX_PARAM_NAME_LENGTH,
    MAX_RECORDS,
    VALID_ACTIONS,
    VALID_REASONS,
    AdaptationCollector,
    AdaptationRecord,
)
from litellm.llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig


# ---------------------------------------------------------------------------
# AdaptationCollector
# ---------------------------------------------------------------------------


class TestAdaptationCollector:
    def test_should_store_value_free_records(self):
        c = AdaptationCollector()
        c.add("temperature", "dropped", "unsupported_param")
        assert c.records == [
            AdaptationRecord("temperature", "dropped", "unsupported_param")
        ]

    def test_should_cap_records_and_count_truncation(self):
        c = AdaptationCollector(max_records=3)
        for i in range(7):
            c.add(f"p{i}", "dropped", "unsupported_param")
        assert len(c) == 3
        assert c.truncated_count == 4

    def test_should_truncate_long_names(self):
        c = AdaptationCollector()
        c.add("x" * 200, "dropped", "unsupported_param")
        assert len(c.records[0].name) == MAX_PARAM_NAME_LENGTH

    def test_should_deduplicate_records(self):
        c = AdaptationCollector()
        c.add("dup", "dropped", "unsupported_param")
        c.add("dup", "dropped", "unsupported_param")
        assert len(c) == 1
        assert c.truncated_count == 0

    def test_should_reject_invalid_max_records(self):
        for bad in (0, -1, "5", None, True, False):
            with pytest.raises(ValueError):
                AdaptationCollector(max_records=bad)

    def test_should_reject_invalid_max_name_length(self):
        for bad in (0, -3, 1.5, True):
            with pytest.raises(ValueError):
                AdaptationCollector(max_name_length=bad)

    def test_should_reject_invalid_action(self):
        c = AdaptationCollector()
        with pytest.raises(ValueError):
            c.add("p", "exploded", "unsupported_param")

    def test_should_reject_invalid_reason(self):
        c = AdaptationCollector()
        with pytest.raises(ValueError):
            c.add("p", "dropped", "because_i_said_so")

    def test_should_expose_typed_literal_sets(self):
        assert VALID_ACTIONS == ("dropped", "rejected")
        assert "unsupported_param" in VALID_REASONS
        assert "extra_body_policy" in VALID_REASONS
        assert "invalid_type" in VALID_REASONS
        assert "non_string_key" not in VALID_REASONS


# ---------------------------------------------------------------------------
# AdaptationRecord frozen dataclass validation
# ---------------------------------------------------------------------------


class TestAdaptationRecord:
    def test_should_reject_invalid_action_at_construction(self):
        with pytest.raises(ValueError, match="action"):
            AdaptationRecord(name="x", action="exploded", reason="unsupported_param")

    def test_should_reject_invalid_reason_at_construction(self):
        with pytest.raises(ValueError, match="reason"):
            AdaptationRecord(name="x", action="dropped", reason="nope")

    def test_should_be_frozen(self):
        rec = AdaptationRecord(name="x", action="dropped", reason="unsupported_param")
        with pytest.raises(AttributeError):
            rec.name = "y"  # type: ignore[misc]

    def test_should_accept_valid_literals(self):
        rec = AdaptationRecord(name="p", action="rejected", reason="invalid_type")
        assert rec.action == "rejected"
        assert rec.reason == "invalid_type"


# ---------------------------------------------------------------------------
# Mapper: allowed fields / snapshot / None / max_tokens
# ---------------------------------------------------------------------------


class TestMapperBasics:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_map_all_allowed_extra_body_fields(self):
        result = self.config.map_openai_params(
            non_default_params={
                "input_type": "query",
                "truncate": "END",
                "modality": "text",
                "embedding_type": "float",
            },
            optional_params={},
        )
        assert result["extra_body"] == {
            "input_type": "query",
            "truncate": "END",
            "modality": "text",
            "embedding_type": "float",
        }

    def test_should_preserve_openai_top_level_and_snapshot(self):
        result = self.config.map_openai_params(
            non_default_params={
                "input_type": "query",
                "dimensions": 256,
                "encoding_format": "base64",
                "user": "u1",
            },
            optional_params={},
        )
        assert result == {
            "extra_body": {"input_type": "query"},
            "dimensions": 256,
            "encoding_format": "base64",
            "user": "u1",
        }

    def test_should_omit_max_tokens(self):
        result = self.config.map_openai_params(
            non_default_params={"max_tokens": 100, "input_type": "passage"},
            optional_params={},
        )
        assert "max_tokens" not in result
        assert "max_tokens" not in result["extra_body"]
        assert result["extra_body"] == {"input_type": "passage"}

    def test_should_ignore_none_without_records(self):
        collector = AdaptationCollector()
        result = self.config.map_openai_params(
            non_default_params={"input_type": None, "temperature": None},
            optional_params={},
            strict=False,
            adaptation_collector=collector,
        )
        assert result["extra_body"] == {}
        assert len(collector) == 0

    def test_should_omit_nested_none_in_extra_body(self):
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"extra_body": {"input_type": "query", "truncate": None}},
            strict=True,
        )
        assert result["extra_body"] == {"input_type": "query"}
        assert "truncate" not in result["extra_body"]


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_raise_unsupported_params_error(self):
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={"temperature": 0.9},
                optional_params={},
                strict=True,
            )

    def test_should_not_leak_values_in_error(self):
        secret = "super-secret-value-12345"
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={"bad_param": secret},
                optional_params={},
                strict=True,
            )
        msg = str(exc_info.value)
        assert secret not in msg
        assert "bad_param" in msg

    def test_should_dedup_and_sort_names(self):
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={"z_param": 1, "a_param": 2, "m_param": 3},
                optional_params={},
                strict=True,
            )
        msg = str(exc_info.value)
        assert msg.index("a_param") < msg.index("z_param")

    def test_should_bound_error_message(self):
        names = {f"param_{i:03d}": i for i in range(MAX_RECORDS + 20)}
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params=names,
                optional_params={},
                strict=True,
            )
        msg = str(exc_info.value)
        assert "more" in msg
        listed = msg.split(": ", 1)[1]
        assert listed.count("param_") <= MAX_RECORDS

    def test_should_truncate_1000_char_name_to_64_in_error(self):
        long_name = "k" * 1000
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={long_name: 1},
                optional_params={},
                strict=True,
            )
        msg = str(exc_info.value)
        assert "k" * MAX_PARAM_NAME_LENGTH in msg
        assert "k" * (MAX_PARAM_NAME_LENGTH + 1) not in msg


# ---------------------------------------------------------------------------
# Drop mode
# ---------------------------------------------------------------------------


class TestDropMode:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_drop_and_record_unsupported(self):
        collector = AdaptationCollector()
        result = self.config.map_openai_params(
            non_default_params={
                "input_type": "query",
                "temperature": 0.5,
                "top_p": 0.9,
            },
            optional_params={},
            strict=False,
            adaptation_collector=collector,
        )
        assert result["extra_body"] == {"input_type": "query"}
        names = {r.name for r in collector.records}
        assert names == {"temperature", "top_p"}
        assert all(r.action == "dropped" for r in collector.records)

    def test_should_truncate_long_names_in_records(self):
        collector = AdaptationCollector()
        self.config.map_openai_params(
            non_default_params={"p" * 200: 1},
            optional_params={},
            strict=False,
            adaptation_collector=collector,
        )
        assert len(collector.records[0].name) == MAX_PARAM_NAME_LENGTH


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


class TestAtomicity:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_not_mutate_optional_params_on_top_level_strict_failure(self):
        optional_params = {"dimensions": 512}
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={"temperature": 0.9},
                optional_params=optional_params,
                strict=True,
            )
        assert optional_params == {"dimensions": 512}

    def test_should_not_mutate_on_nested_extra_body_strict_failure(self):
        optional_params = {"extra_body": {"input_type": "query"}}
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={},
                optional_params=optional_params,
                kwargs={"extra_body": {"api_key": "sk-secret"}},
                strict=True,
            )
        assert optional_params == {"extra_body": {"input_type": "query"}}

    def test_should_reject_preexisting_bad_extra_body_key_strict(self):
        optional_params = {"extra_body": {"api_key": "sk-x"}}
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={"input_type": "query"},
                optional_params=optional_params,
                strict=True,
            )
        assert "api_key" in str(exc_info.value)
        assert optional_params == {"extra_body": {"api_key": "sk-x"}}

    def test_should_strip_preexisting_bad_key_in_drop_mode(self):
        collector = AdaptationCollector()
        optional_params = {"extra_body": {"api_key": "sk-x", "input_type": "query"}}
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params=optional_params,
            strict=False,
            adaptation_collector=collector,
        )
        assert "api_key" not in result["extra_body"]
        assert result["extra_body"]["input_type"] == "query"

    def test_should_drop_preexisting_integer_key_and_not_leak_value(self):
        collector = AdaptationCollector()
        secret_val = "internal-secret-999"
        optional_params = {"extra_body": {42: secret_val, "input_type": "query"}}
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params=optional_params,
            strict=False,
            adaptation_collector=collector,
        )
        assert 42 not in result["extra_body"]
        assert result["extra_body"] == {"input_type": "query"}
        names = {r.name for r in collector.records}
        assert "42" in names
        # Value must not appear in any record field.
        for rec in collector.records:
            assert secret_val not in (rec.name, rec.action, rec.reason)

    def test_should_reject_preexisting_integer_key_strict_atomic(self):
        optional_params = {"extra_body": {42: "val", "input_type": "query"}}
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={},
                optional_params=optional_params,
                strict=True,
            )
        assert "42" in str(exc_info.value)
        # Atomic: no mutation.
        assert optional_params == {"extra_body": {42: "val", "input_type": "query"}}


# ---------------------------------------------------------------------------
# Preexisting extra_body non-dict
# ---------------------------------------------------------------------------


class TestPreexistingExtraBodyNonDict:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_raise_on_preexisting_non_dict_extra_body_strict(self):
        optional_params = {"extra_body": "garbage"}
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={"input_type": "query"},
                optional_params=optional_params,
                strict=True,
            )
        assert "extra_body" in str(exc_info.value)
        # Atomic: no mutation.
        assert optional_params == {"extra_body": "garbage"}

    def test_should_replace_non_dict_extra_body_in_drop_mode(self):
        collector = AdaptationCollector()
        optional_params = {"extra_body": [1, 2, 3]}
        result = self.config.map_openai_params(
            non_default_params={"input_type": "query"},
            optional_params=optional_params,
            strict=False,
            adaptation_collector=collector,
        )
        assert result["extra_body"] == {"input_type": "query"}
        recs = [r for r in collector.records if r.name == "extra_body"]
        assert len(recs) == 1
        assert recs[0].action == "rejected"
        assert recs[0].reason == "invalid_type"


# ---------------------------------------------------------------------------
# Explicit extra_body policy
# ---------------------------------------------------------------------------


class TestExplicitExtraBody:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_merge_valid_extra_body(self):
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"extra_body": {"modality": "text"}},
            strict=True,
        )
        assert result["extra_body"]["modality"] == "text"

    def test_should_reject_non_dict_extra_body_strict(self):
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={},
                optional_params={},
                kwargs={"extra_body": "not-a-dict"},
                strict=True,
            )

    def test_should_reject_collision_with_mapped_field(self):
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={"input_type": "query"},
                optional_params={},
                kwargs={"extra_body": {"input_type": "passage"}},
                strict=True,
            )

    def test_should_copy_not_reference_extra_body(self):
        original = {"truncate": "END"}
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"extra_body": original},
            strict=True,
        )
        result["extra_body"]["truncate"] = "NONE"
        assert original["truncate"] == "END"

    def test_should_record_invalid_type_in_drop_mode(self):
        collector = AdaptationCollector()
        self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"extra_body": [1, 2, 3]},
            strict=False,
            adaptation_collector=collector,
        )
        assert collector.records[0].reason == "invalid_type"


# ---------------------------------------------------------------------------
# Non-string / mixed keys
# ---------------------------------------------------------------------------


class TestNonStringKeys:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_handle_non_string_top_level_key_strict(self):
        with pytest.raises(UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={123: "value"},
                optional_params={},
                strict=True,
            )
        assert "123" in str(exc_info.value)

    def test_should_handle_non_string_extra_body_key_drop(self):
        collector = AdaptationCollector()
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"extra_body": {999: "value"}},
            strict=False,
            adaptation_collector=collector,
        )
        assert 999 not in result["extra_body"]
        assert collector.records[0].name == "999"

    def test_should_not_raise_typeerror_on_mixed_keys(self):
        collector = AdaptationCollector()
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"extra_body": {1: "a", "b": "c", 2: "d"}},
            strict=False,
            adaptation_collector=collector,
        )
        assert isinstance(result["extra_body"], dict)


# ---------------------------------------------------------------------------
# Compatibility default (existing direct callers)
# ---------------------------------------------------------------------------


class TestCompatibilityDefault:
    def setup_method(self):
        self.config = NvidiaNimEmbeddingConfig()

    def test_should_preserve_legacy_forwarding_without_policy_args(self):
        result = self.config.map_openai_params(
            non_default_params={"input_type": "passage", "dimensions": 1024},
            optional_params={},
            kwargs={"some_unknown_kwarg": "x"},
        )
        assert result["extra_body"] == {
            "input_type": "passage",
            "some_unknown_kwarg": "x",
        }
        assert result["dimensions"] == 1024

    def test_should_activate_policy_when_collector_provided(self):
        collector = AdaptationCollector()
        result = self.config.map_openai_params(
            non_default_params={"input_type": "passage"},
            optional_params={},
            kwargs={"some_unknown_kwarg": "x"},
            adaptation_collector=collector,
        )
        assert result["extra_body"] == {"input_type": "passage"}
        assert "some_unknown_kwarg" not in result["extra_body"]
        assert {r.name for r in collector.records} == {"some_unknown_kwarg"}


# ---------------------------------------------------------------------------
# D1-541 compatibility smoke (single narrow test)
# ---------------------------------------------------------------------------


class TestD1541Compat:
    def test_should_keep_instance_state_isolated(self):
        a = NvidiaNimEmbeddingConfig(input_type="query")
        b = NvidiaNimEmbeddingConfig(truncate="END")
        assert a._get_instance_config() == {"input_type": "query"}
        assert b._get_instance_config() == {"truncate": "END"}
        assert NvidiaNimEmbeddingConfig.get_config() == {}
