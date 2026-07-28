import json
import threading
from pathlib import Path

import pytest

import litellm

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.param_adaptation import AdaptationCollector
from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig


REPO_ROOT = Path(__file__).resolve().parents[5]
NVIDIA_ADAPTER_COST_MAP_MODELS = {
    "nvidia_nim/deepseek-ai/deepseek-v3.1-terminus",
    "nvidia_nim/deepseek-ai/deepseek-v3.2",
    "nvidia_nim/minimaxai/minimax-m2.7",
    "nvidia_nim/mistralai/devstral-2-123b-instruct-2512",
    "nvidia_nim/z-ai/glm4.7",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mct_resolver(model: str, *, supported: bool):
    """Build an instance-local resolver reporting *supported* for *model*.

    No global / class / catalog mutation: the returned callable closes over
    a private dict, so each config instance is fully isolated.
    """
    table = {
        model: {
            "supports_max_completion_tokens": supported,
        }
    }

    def resolver(m, provider):
        return table.get(m)

    return resolver


def _synthetic_mct_entry(*, supported: bool) -> dict:
    """Minimal model_cost entry exercising the typed helper path."""
    return {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "nvidia_nim",
        "mode": "chat",
        "supports_max_completion_tokens": supported,
    }


# ---------------------------------------------------------------------------
# Token-policy: conservative / max_tokens-only family (no metadata)
# ---------------------------------------------------------------------------


class TestTokenPolicyConservative:
    """Default path when model metadata lacks ``supports_max_completion_tokens``."""

    def test_only_max_tokens_passthrough(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 1024},
            optional_params={},
            model="vendor/unknown-model",
            drop_params=False,
        )
        assert result == {"max_tokens": 1024}

    def test_only_max_completion_tokens_mapped_to_max_tokens(self) -> None:
        """Historical NVIDIA compat: alias maps to provider-native field."""
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 2048},
            optional_params={},
            model="vendor/unknown-model",
            drop_params=False,
        )
        assert result == {"max_tokens": 2048}
        assert "max_completion_tokens" not in result

    def test_equal_aliases_deduplicate_to_max_tokens(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "max_tokens": 512,
                "max_completion_tokens": 512,
            },
            optional_params={},
            model="vendor/unknown-model",
            drop_params=False,
        )
        assert result == {"max_tokens": 512}

    def test_conflicting_aliases_strict_raises_names_only(self) -> None:
        config = NvidiaNimConfig()
        with pytest.raises(UnsupportedParamsError) as exc_info:
            config.map_openai_params(
                non_default_params={
                    "max_tokens": 100,
                    "max_completion_tokens": 200,
                },
                optional_params={},
                model="vendor/unknown-model",
                drop_params=False,
            )
        msg = str(exc_info.value.message)
        assert "max_tokens" in msg
        assert "max_completion_tokens" in msg
        # Names-only: no parameter values leaked.
        assert "100" not in msg
        assert "200" not in msg

    def test_conflicting_aliases_drop_retains_native_and_records(self) -> None:
        config = NvidiaNimConfig()
        collector = AdaptationCollector()
        result = config.map_openai_params(
            non_default_params={
                "max_tokens": 100,
                "max_completion_tokens": 200,
            },
            optional_params={},
            model="vendor/unknown-model",
            drop_params=True,
            adaptation_collector=collector,
        )
        assert result == {"max_tokens": 100}
        assert "max_completion_tokens" not in result
        # Value-free adaptation recorded.
        assert len(collector) == 1
        rec = collector.records[0]
        assert rec.name == "max_completion_tokens"
        assert rec.action == "dropped"
        assert rec.reason == "unsupported_param"

    def test_conflicting_aliases_drop_without_collector(self) -> None:
        """No collector supplied: still deterministic, no error."""
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "max_tokens": 300,
                "max_completion_tokens": 400,
            },
            optional_params={},
            model="vendor/unknown-model",
            drop_params=True,
        )
        assert result == {"max_tokens": 300}


# ---------------------------------------------------------------------------
# Token-policy: native max_completion_tokens family (metadata-driven)
# ---------------------------------------------------------------------------


class TestTokenPolicyNativeMCT:
    """When metadata declares ``supports_max_completion_tokens: true``.

    Uses instance-local resolver injection (no global/class mutation).
    """

    def _config(self) -> NvidiaNimConfig:
        return NvidiaNimConfig(
            metadata_resolver=_mct_resolver("vendor/mct-model", supported=True)
        )

    def test_only_alias_preserved_native(self) -> None:
        config = self._config()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 4096},
            optional_params={},
            model="vendor/mct-model",
            drop_params=False,
        )
        assert result == {"max_completion_tokens": 4096}
        assert "max_tokens" not in result

    def test_only_native_passthrough(self) -> None:
        config = self._config()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 8192},
            optional_params={},
            model="vendor/mct-model",
            drop_params=False,
        )
        assert result == {"max_tokens": 8192}

    def test_equal_aliases_deduplicate_to_native_field(self) -> None:
        config = self._config()
        result = config.map_openai_params(
            non_default_params={
                "max_tokens": 512,
                "max_completion_tokens": 512,
            },
            optional_params={},
            model="vendor/mct-model",
            drop_params=False,
        )
        # Equal aliases always deduplicate to provider-native field.
        assert result == {"max_tokens": 512}

    def test_conflicting_aliases_strict_raises(self) -> None:
        config = self._config()
        with pytest.raises(UnsupportedParamsError) as exc_info:
            config.map_openai_params(
                non_default_params={
                    "max_tokens": 100,
                    "max_completion_tokens": 200,
                },
                optional_params={},
                model="vendor/mct-model",
                drop_params=False,
            )
        msg = str(exc_info.value.message)
        assert "max_tokens" in msg
        assert "max_completion_tokens" in msg
        assert "100" not in msg
        assert "200" not in msg

    def test_conflicting_aliases_drop_retains_native_records(self) -> None:
        config = self._config()
        collector = AdaptationCollector()
        result = config.map_openai_params(
            non_default_params={
                "max_tokens": 100,
                "max_completion_tokens": 200,
            },
            optional_params={},
            model="vendor/mct-model",
            drop_params=True,
            adaptation_collector=collector,
        )
        assert result == {"max_tokens": 100}
        assert "max_completion_tokens" not in result
        assert len(collector) == 1
        rec = collector.records[0]
        assert rec.name == "max_completion_tokens"
        assert rec.action == "dropped"
        assert rec.reason == "unsupported_param"


# ---------------------------------------------------------------------------
# Typed helper integration: prefixed/unprefixed synthetic true/false/absent
# ---------------------------------------------------------------------------


class TestTypedHelperIntegration:
    """Prove the default resolver uses get_nvidia_nim_model_metadata.

    These tests exercise the real typed helper path by injecting synthetic
    model_cost entries (monkeypatch auto-reverts).  No class/global resolver
    mutation.
    """

    def test_unprefixed_synthetic_true(self, monkeypatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-true",
            _synthetic_mct_entry(supported=True),
        )
        config = NvidiaNimConfig()
        assert config._provider_supports_max_completion_tokens("vendor/synth-true")

    def test_prefixed_synthetic_true(self, monkeypatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-true",
            _synthetic_mct_entry(supported=True),
        )
        config = NvidiaNimConfig()
        assert config._provider_supports_max_completion_tokens(
            "nvidia_nim/vendor/synth-true"
        )

    def test_unprefixed_synthetic_false(self, monkeypatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-false",
            _synthetic_mct_entry(supported=False),
        )
        config = NvidiaNimConfig()
        assert not config._provider_supports_max_completion_tokens("vendor/synth-false")

    def test_prefixed_synthetic_false(self, monkeypatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-false",
            _synthetic_mct_entry(supported=False),
        )
        config = NvidiaNimConfig()
        assert not config._provider_supports_max_completion_tokens(
            "nvidia_nim/vendor/synth-false"
        )

    def test_absent_model_returns_false(self) -> None:
        config = NvidiaNimConfig()
        assert not config._provider_supports_max_completion_tokens(
            "totally/nonexistent-model-xyz"
        )

    def test_prefixed_and_unprefixed_identical(self, monkeypatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-true",
            _synthetic_mct_entry(supported=True),
        )
        config = NvidiaNimConfig()
        unprefixed = config._provider_supports_max_completion_tokens(
            "vendor/synth-true"
        )
        prefixed = config._provider_supports_max_completion_tokens(
            "nvidia_nim/vendor/synth-true"
        )
        assert unprefixed == prefixed is True

    def test_synthetic_true_drives_native_alias_passthrough(self, monkeypatch) -> None:
        """End-to-end: synthetic true metadata -> alias preserved natively."""
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-true",
            _synthetic_mct_entry(supported=True),
        )
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 7777},
            optional_params={},
            model="vendor/synth-true",
            drop_params=False,
        )
        assert result == {"max_completion_tokens": 7777}

    def test_synthetic_false_drives_conservative_mapping(self, monkeypatch) -> None:
        """End-to-end: synthetic false metadata -> alias mapped to max_tokens."""
        monkeypatch.setitem(
            litellm.model_cost,
            "nvidia_nim/vendor/synth-false",
            _synthetic_mct_entry(supported=False),
        )
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 7777},
            optional_params={},
            model="vendor/synth-false",
            drop_params=False,
        )
        assert result == {"max_tokens": 7777}


# ---------------------------------------------------------------------------
# Resolver isolation: sequential + thread interleaving
# ---------------------------------------------------------------------------


class TestResolverIsolation:
    """Instance-local resolver must not leak across instances/threads."""

    def test_sequential_instances_isolated(self) -> None:
        native = NvidiaNimConfig(
            metadata_resolver=_mct_resolver("vendor/mct-model", supported=True)
        )
        conservative = NvidiaNimConfig()

        assert native.map_openai_params(
            non_default_params={"max_completion_tokens": 4096},
            optional_params={},
            model="vendor/mct-model",
            drop_params=False,
        ) == {"max_completion_tokens": 4096}

        # A default instance observing the same model must stay conservative.
        assert conservative.map_openai_params(
            non_default_params={"max_completion_tokens": 4096},
            optional_params={},
            model="vendor/mct-model",
            drop_params=False,
        ) == {"max_tokens": 4096}

        # Re-check native after conservative use (no cross-contamination).
        assert native.map_openai_params(
            non_default_params={"max_completion_tokens": 4096},
            optional_params={},
            model="vendor/mct-model",
            drop_params=False,
        ) == {"max_completion_tokens": 4096}

    def test_thread_isolation_under_interleaving(self) -> None:
        native = NvidiaNimConfig(
            metadata_resolver=_mct_resolver("vendor/mct-model", supported=True)
        )
        conservative = NvidiaNimConfig()

        barrier = threading.Barrier(2)
        errors: list = []
        native_results: list = []
        conservative_results: list = []

        def run_native() -> None:
            try:
                barrier.wait()
                for _ in range(200):
                    native_results.append(
                        native.map_openai_params(
                            non_default_params={"max_completion_tokens": 4096},
                            optional_params={},
                            model="vendor/mct-model",
                            drop_params=False,
                        )
                    )
            except Exception as exc:  # pragma: no cover - surfaced via errors
                errors.append(("native", exc))

        def run_conservative() -> None:
            try:
                barrier.wait()
                for _ in range(200):
                    conservative_results.append(
                        conservative.map_openai_params(
                            non_default_params={"max_completion_tokens": 4096},
                            optional_params={},
                            model="vendor/mct-model",
                            drop_params=False,
                        )
                    )
            except Exception as exc:  # pragma: no cover - surfaced via errors
                errors.append(("conservative", exc))

        t1 = threading.Thread(target=run_native)
        t2 = threading.Thread(target=run_conservative)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert len(native_results) == 200
        assert len(conservative_results) == 200
        # Every native result keeps the alias; every conservative maps it.
        assert all(r == {"max_completion_tokens": 4096} for r in native_results)
        assert all(r == {"max_tokens": 4096} for r in conservative_results)

    def test_default_construction_has_no_instance_resolver_leak(self) -> None:
        """Constructing an injected config must not alter a default one."""
        _ = NvidiaNimConfig(
            metadata_resolver=_mct_resolver("vendor/mct-model", supported=True)
        )
        default = NvidiaNimConfig()
        assert not default._provider_supports_max_completion_tokens("vendor/mct-model")


# ---------------------------------------------------------------------------
# Stale optional_params cleanup
# ---------------------------------------------------------------------------


class TestStaleParamCleanup:
    """Pre-existing stale token-limit keys in optional_params are removed."""

    def test_stale_alias_removed_on_conservative_resolve(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={"max_completion_tokens": 999},
            model="vendor/unknown-model",
            drop_params=False,
        )
        assert result == {"max_tokens": 256}
        assert "max_completion_tokens" not in result

    def test_stale_native_removed_on_alias_resolve(self) -> None:
        config = NvidiaNimConfig(
            metadata_resolver=_mct_resolver("vendor/mct-model", supported=True)
        )
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 4096},
            optional_params={"max_tokens": 1},
            model="vendor/mct-model",
            drop_params=False,
        )
        assert result == {"max_completion_tokens": 4096}
        assert "max_tokens" not in result

    def test_both_stale_cleared_on_dedup(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "max_tokens": 512,
                "max_completion_tokens": 512,
            },
            optional_params={"max_tokens": 1, "max_completion_tokens": 2},
            model="vendor/unknown-model",
            drop_params=False,
        )
        assert result == {"max_tokens": 512}


# ---------------------------------------------------------------------------
# Unknown model: conservative behavior (consolidated)
# ---------------------------------------------------------------------------


class TestUnknownModelConservative:
    """Unknown models (no metadata) must default to max_tokens-only."""

    def test_alias_mapped_and_conflict_strict(self) -> None:
        config = NvidiaNimConfig()
        # Alias maps to max_tokens for unknown model.
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 999},
            optional_params={},
            model="totally/unknown-model-xyz",
            drop_params=False,
        )
        assert result == {"max_tokens": 999}
        # Conflict raises for unknown model.
        with pytest.raises(UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={
                    "max_tokens": 1,
                    "max_completion_tokens": 2,
                },
                optional_params={},
                model="totally/unknown-model-xyz",
                drop_params=False,
            )


# ---------------------------------------------------------------------------
# Unrelated chat mapping preserved
# ---------------------------------------------------------------------------


class TestUnrelatedMappingPreserved:
    """Ensure non-token params are unaffected by the token-limit refactor."""

    def test_standard_params_pass_through(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["END"],
                "seed": 42,
            },
            optional_params={},
            model="meta/llama3-8b-instruct",
            drop_params=False,
        )
        assert result == {
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["END"],
            "seed": 42,
        }

    def test_unsupported_param_dropped(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "temperature": 0.5,
                "cache_control": {"type": "ephemeral"},
            },
            optional_params={},
            model="meta/llama3-8b-instruct",
            drop_params=False,
        )
        assert result == {"temperature": 0.5}
        assert "cache_control" not in result

    def test_tools_and_response_format_preserved(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": "auto",
                "response_format": {"type": "json_object"},
            },
            optional_params={},
            model="meta/llama3-8b-instruct",
            drop_params=False,
        )
        assert "tools" in result
        assert result["tool_choice"] == "auto"
        assert result["response_format"] == {"type": "json_object"}

    def test_token_and_non_token_combined(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={
                "max_completion_tokens": 256,
                "temperature": 0.1,
            },
            optional_params={},
            model="meta/llama3-8b-instruct",
            drop_params=False,
        )
        assert result == {"max_tokens": 256, "temperature": 0.1}

    def test_streaming_param_preserved(self) -> None:
        config = NvidiaNimConfig()
        result = config.map_openai_params(
            non_default_params={"stream": True, "max_tokens": 128},
            optional_params={},
            model="meta/llama3-8b-instruct",
            drop_params=False,
        )
        assert result == {"stream": True, "max_tokens": 128}


# ---------------------------------------------------------------------------
# Adaptation metadata serialization
# ---------------------------------------------------------------------------


class TestAdaptationMetadata:
    """Collector metadata is deterministic and value-free."""

    def test_metadata_shape(self) -> None:
        config = NvidiaNimConfig()
        collector = AdaptationCollector()
        config.map_openai_params(
            non_default_params={
                "max_tokens": 10,
                "max_completion_tokens": 20,
            },
            optional_params={},
            model="vendor/unknown-model",
            drop_params=True,
            adaptation_collector=collector,
        )
        meta = collector.to_metadata()
        records = meta["provider_parameter_adaptations"]
        assert len(records) == 1
        assert records[0] == {
            "name": "max_completion_tokens",
            "action": "dropped",
            "reason": "unsupported_param",
        }
        assert meta["provider_parameter_adaptations_truncated_count"] == 0


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_nvidia_nim_reasoning_effort_supported_from_model_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "nvidia_nim/vendor/reasoning-model",
        {
            "input_cost_per_token": 0.0,
            "litellm_provider": "nvidia_nim",
            "mode": "chat",
            "output_cost_per_token": 0.0,
            "supports_reasoning": True,
        },
    )

    config = NvidiaNimConfig()

    supported_params = config.get_supported_openai_params("vendor/reasoning-model")
    optional_params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": "high",
            "cache_control": {"type": "ephemeral"},
        },
        optional_params={},
        model="vendor/reasoning-model",
        drop_params=False,
    )

    assert "reasoning_effort" in supported_params
    assert optional_params["reasoning_effort"] == "high"
    assert "cache_control" not in optional_params


def test_nvidia_nim_reasoning_effort_stripped_without_model_metadata() -> None:
    config = NvidiaNimConfig()

    supported_params = config.get_supported_openai_params("vendor/plain-model")
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="vendor/plain-model",
        drop_params=False,
    )

    assert "reasoning_effort" not in supported_params
    assert "reasoning_effort" not in optional_params


def test_nvidia_anthropic_adapter_models_have_nonzero_cost_map_coverage() -> None:
    for path in (
        REPO_ROOT / "model_prices_and_context_window.json",
        REPO_ROOT / "litellm" / "bundled_model_prices_and_context_window_fallback.json",
    ):
        model_cost = json.loads(path.read_text())
        for model in NVIDIA_ADAPTER_COST_MAP_MODELS:
            entry = model_cost[model]
            assert entry["litellm_provider"] == "nvidia_nim"
            assert entry["mode"] == "chat"
            assert entry["input_cost_per_token"] > 0
            assert entry["output_cost_per_token"] > 0
            assert entry["supports_function_calling"] is True
            assert entry["supports_tool_choice"] is True


def test_nvidia_minimax_cost_map_uses_openrouter_fallback_pricing_basis() -> None:
    for path in (
        REPO_ROOT / "model_prices_and_context_window.json",
        REPO_ROOT / "litellm" / "bundled_model_prices_and_context_window_fallback.json",
    ):
        entry = json.loads(path.read_text())["nvidia_nim/minimaxai/minimax-m2.7"]
        assert entry["pricing_source_model"] == "openrouter/minimax/minimax-m2.5"
        assert entry["pricing_source"] == "https://openrouter.ai/minimax/minimax-m2.5"
        assert entry["input_cost_per_token"] == 1.5e-07
        assert entry["output_cost_per_token"] == 1.15e-06
