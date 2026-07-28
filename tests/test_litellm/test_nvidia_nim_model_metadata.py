import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
from typing_extensions import assert_type

import litellm
from litellm.llms.nvidia_nim.rerank.ranking_transformation import (
    NvidiaNimRankingConfig,
)
from litellm.llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig
from litellm.types.utils import (
    NvidiaNimModelMetadata,
    NvidiaNimRerankMetadata,
    get_nvidia_nim_model_metadata,
)
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,
    get_model_info,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATHS = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "bundled_model_prices_and_context_window_fallback.json",
)

CatalogEntry = Dict[str, Any]
Catalog = Dict[str, CatalogEntry]

SYNTHETIC_CHAT_CAPABILITY_CASES: tuple[tuple[str, Optional[bool]], ...] = (
    ("nvidia_nim/test/typed-max-completion-true", True),
    ("nvidia_nim/test/typed-max-completion-false", False),
    ("nvidia_nim/test/typed-max-completion-absent", None),
)


def _load_catalog(path: Path) -> Catalog:
    return json.loads(path.read_text(encoding="utf-8"))


def _nvidia_catalog_entries(catalog: Catalog, mode: str) -> Dict[str, CatalogEntry]:
    return {
        model: entry
        for model, entry in catalog.items()
        if entry.get("litellm_provider") == "nvidia_nim" and entry.get("mode") == mode
    }


def _rerank_metadata(entry: CatalogEntry) -> NvidiaNimRerankMetadata:
    provider_specific_entry = entry.get("provider_specific_entry")
    assert isinstance(provider_specific_entry, dict)
    assert set(provider_specific_entry) == {"rerank"}

    rerank_metadata = provider_specific_entry["rerank"]
    assert isinstance(rerank_metadata, dict)
    assert set(rerank_metadata) == {"body_model", "endpoint_path"}
    assert isinstance(rerank_metadata["body_model"], str)
    assert isinstance(rerank_metadata["endpoint_path"], str)
    return NvidiaNimRerankMetadata(
        body_model=rerank_metadata["body_model"],
        endpoint_path=rerank_metadata["endpoint_path"],
    )


def _rerank_config(model: str) -> NvidiaNimRerankConfig:
    if model.removeprefix("nvidia_nim/").startswith("ranking/"):
        return NvidiaNimRankingConfig()
    return NvidiaNimRerankConfig()


def _clear_model_info_caches() -> None:
    _invalidate_model_cost_lowercase_map()
    get_model_info.cache_clear()


@pytest.fixture
def local_nvidia_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    catalog = _load_catalog(CATALOG_PATHS[0])
    for model, supports_max_completion_tokens in SYNTHETIC_CHAT_CAPABILITY_CASES:
        entry: CatalogEntry = {
            "input_cost_per_token": 0.0,
            "litellm_provider": "nvidia_nim",
            "mode": "chat",
            "output_cost_per_token": 0.0,
        }
        if supports_max_completion_tokens is not None:
            entry["supports_max_completion_tokens"] = supports_max_completion_tokens
        catalog[model] = entry

    monkeypatch.setattr(litellm, "model_cost", catalog)
    _clear_model_info_caches()
    yield
    _clear_model_info_caches()


def test_should_keep_catalog_copies_byte_identical_and_valid() -> None:
    catalog_texts = [
        catalog_path.read_text(encoding="utf-8") for catalog_path in CATALOG_PATHS
    ]

    assert catalog_texts[0] == catalog_texts[1]
    assert json.loads(catalog_texts[0]) == json.loads(catalog_texts[1])


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_preserve_exact_nvidia_rerank_catalog_contracts() -> None:
    endpoint_families = set()

    for catalog_path in CATALOG_PATHS:
        catalog = _load_catalog(catalog_path)
        nvidia_rerank_entries = _nvidia_catalog_entries(catalog, mode="rerank")
        assert nvidia_rerank_entries

        for catalog_model, entry in nvidia_rerank_entries.items():
            expected_rerank = _rerank_metadata(entry)
            unprefixed_model = catalog_model.removeprefix("nvidia_nim/")
            config = _rerank_config(catalog_model)

            for model in (catalog_model, unprefixed_model):
                assert (
                    config._get_model_name_for_body(model)
                    == expected_rerank["body_model"]
                )
                runtime_url = config.get_complete_url(None, model)
                assert runtime_url.startswith(config.DEFAULT_NIM_RERANK_API_BASE)
                runtime_path = "/" + runtime_url.removeprefix(
                    config.DEFAULT_NIM_RERANK_API_BASE
                ).lstrip("/")
                assert runtime_path == expected_rerank["endpoint_path"]

            endpoint_path = expected_rerank["endpoint_path"]
            if endpoint_path == "/v1/ranking":
                endpoint_families.add("/v1/ranking")
            elif endpoint_path.startswith("/v1/retrieval/"):
                endpoint_families.add("/v1/retrieval")
            else:
                pytest.fail(f"Unexpected NVIDIA NIM rerank endpoint: {endpoint_path}")

    assert endpoint_families == {"/v1/ranking", "/v1/retrieval"}


def test_should_keep_current_nvidia_chat_models_conservative() -> None:
    for catalog_path in CATALOG_PATHS:
        catalog = _load_catalog(catalog_path)
        nvidia_chat_entries = _nvidia_catalog_entries(catalog, mode="chat")

        assert nvidia_chat_entries
        for entry in nvidia_chat_entries.values():
            provider_specific_entry = entry.get("provider_specific_entry") or {}
            assert entry.get("supports_max_completion_tokens") is not True
            assert (
                provider_specific_entry.get("supports_max_completion_tokens")
                is not True
            )


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_normalize_prefixed_and_unprefixed_rerank_metadata() -> None:
    catalog = _load_catalog(CATALOG_PATHS[0])
    for catalog_model, entry in _nvidia_catalog_entries(catalog, mode="rerank").items():
        expected_rerank = _rerank_metadata(entry)
        unprefixed_model = catalog_model.removeprefix("nvidia_nim/")
        expected_metadata = NvidiaNimModelMetadata(rerank=expected_rerank)

        assert get_nvidia_nim_model_metadata(catalog_model) == expected_metadata
        assert get_nvidia_nim_model_metadata(unprefixed_model) == expected_metadata

        prefixed_info = get_model_info(
            catalog_model,
            custom_llm_provider="nvidia_nim",
        )
        unprefixed_info = get_model_info(
            unprefixed_model,
            custom_llm_provider="nvidia_nim",
        )
        assert prefixed_info["key"] == catalog_model
        assert unprefixed_info["key"] == catalog_model
        assert prefixed_info["provider_specific_entry"] == {"rerank": expected_rerank}
        assert unprefixed_info["provider_specific_entry"] == {"rerank": expected_rerank}


@pytest.mark.parametrize(
    ("catalog_model", "expected_support"),
    SYNTHETIC_CHAT_CAPABILITY_CASES,
)
@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_propagate_synthetic_chat_capability(
    catalog_model: str,
    expected_support: Optional[bool],
) -> None:
    unprefixed_model = catalog_model.removeprefix("nvidia_nim/")
    for model in (catalog_model, unprefixed_model):
        model_info = get_model_info(
            model,
            custom_llm_provider="nvidia_nim",
        )
        assert_type(
            model_info.get("supports_max_completion_tokens"),
            Optional[bool],
        )
        assert model_info.get("supports_max_completion_tokens") is expected_support

        metadata = get_nvidia_nim_model_metadata(model)
        if expected_support is None:
            assert "supports_max_completion_tokens" not in metadata
            assert metadata.get("supports_max_completion_tokens", False) is False
        else:
            assert metadata["supports_max_completion_tokens"] is expected_support


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_default_current_chat_capability_to_false() -> None:
    catalog = _load_catalog(CATALOG_PATHS[0])
    for catalog_model in _nvidia_catalog_entries(catalog, mode="chat"):
        unprefixed_model = catalog_model.removeprefix("nvidia_nim/")
        for model in (catalog_model, unprefixed_model):
            model_info = get_model_info(
                model,
                custom_llm_provider="nvidia_nim",
            )
            assert_type(
                model_info.get("supports_max_completion_tokens"),
                Optional[bool],
            )
            assert model_info.get("supports_max_completion_tokens") is None

            metadata = get_nvidia_nim_model_metadata(model)
            assert "supports_max_completion_tokens" not in metadata
            assert metadata.get("supports_max_completion_tokens", False) is False


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_return_empty_metadata_for_unknown_models() -> None:
    assert get_nvidia_nim_model_metadata("vendor/not-in-catalog") == {}
    assert get_nvidia_nim_model_metadata("nvidia_nim/vendor/not-in-catalog") == {}


def test_should_expose_typed_nvidia_metadata() -> None:
    rerank_metadata = NvidiaNimRerankMetadata(
        endpoint_path="/v1/ranking",
        body_model="nvidia/example-reranker",
    )
    metadata = NvidiaNimModelMetadata(
        supports_max_completion_tokens=True,
        rerank=rerank_metadata,
    )

    assert_type(
        metadata.get("supports_max_completion_tokens"),
        Optional[bool],
    )
    assert_type(metadata["rerank"], NvidiaNimRerankMetadata)
    assert metadata["supports_max_completion_tokens"] is True


# ---------------------------------------------------------------------------
# D1-556: Typed NVIDIA catalog authority hardening
# ---------------------------------------------------------------------------

# The exact five NVIDIA NIM rerank catalog keys and their typed metadata.
EXPECTED_NVIDIA_RERANK_CATALOG: Dict[str, NvidiaNimRerankMetadata] = {
    "nvidia_nim/nv-rerank-qa-mistral-4b:1": NvidiaNimRerankMetadata(
        endpoint_path="/v1/retrieval/nvidia/reranking",
        body_model="nv-rerank-qa-mistral-4b:1",
    ),
    "nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2": NvidiaNimRerankMetadata(
        endpoint_path="/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        body_model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ),
    "nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3": NvidiaNimRerankMetadata(
        endpoint_path="/v1/retrieval/nvidia/reranking",
        body_model="nvidia/nv-rerankqa-mistral-4b-v3",
    ),
    "nvidia_nim/nvidia/rerank-qa-mistral-4b": NvidiaNimRerankMetadata(
        endpoint_path="/v1/retrieval/nvidia/reranking",
        body_model="nv-rerank-qa-mistral-4b:1",
    ),
    "nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2": NvidiaNimRerankMetadata(
        endpoint_path="/v1/ranking",
        body_model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ),
}


def test_should_have_exactly_five_nvidia_rerank_catalog_keys() -> None:
    """Assert the exact set of NVIDIA rerank keys in both catalogs."""
    for catalog_path in CATALOG_PATHS:
        catalog = _load_catalog(catalog_path)
        nvidia_rerank_entries = _nvidia_catalog_entries(catalog, mode="rerank")
        assert set(nvidia_rerank_entries.keys()) == set(
            EXPECTED_NVIDIA_RERANK_CATALOG.keys()
        ), (
            f"Catalog {catalog_path.name}: expected exactly "
            f"{sorted(EXPECTED_NVIDIA_RERANK_CATALOG.keys())}, "
            f"got {sorted(nvidia_rerank_entries.keys())}"
        )


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_resolve_exact_typed_shape_for_each_rerank_key() -> None:
    """Each known rerank key resolves to exact endpoint_path/body_model."""
    for catalog_model, expected_rerank in EXPECTED_NVIDIA_RERANK_CATALOG.items():
        metadata = get_nvidia_nim_model_metadata(catalog_model)
        assert metadata.get("rerank") == expected_rerank, (
            f"Model {catalog_model}: expected {expected_rerank}, "
            f"got {metadata.get('rerank')}"
        )
        # Verify typed shape explicitly
        rerank = metadata["rerank"]
        assert_type(rerank, NvidiaNimRerankMetadata)
        assert isinstance(rerank["endpoint_path"], str)
        assert isinstance(rerank["body_model"], str)


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_return_empty_for_genuine_unknown_model() -> None:
    """Unknown models produce empty metadata (ValueError swallowed)."""
    result = get_nvidia_nim_model_metadata("vendor/definitely-not-in-catalog")
    assert result == {}
    assert isinstance(result, dict)


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_raise_typeerror_for_malformed_provider_specific_entry() -> None:
    """Non-Mapping provider_specific_entry raises TypeError."""
    catalog_model = next(iter(EXPECTED_NVIDIA_RERANK_CATALOG))
    fake_model_info: Dict[str, Any] = {
        "key": catalog_model,
        "litellm_provider": "nvidia_nim",
        "mode": "rerank",
        "provider_specific_entry": "not-a-mapping",
    }
    with patch("litellm.utils.get_model_info", return_value=fake_model_info):
        with pytest.raises(TypeError, match="Malformed provider_specific_entry"):
            get_nvidia_nim_model_metadata(catalog_model)


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_raise_typeerror_for_malformed_rerank_subentry() -> None:
    """Non-Mapping rerank sub-entry raises TypeError."""
    catalog_model = next(iter(EXPECTED_NVIDIA_RERANK_CATALOG))
    fake_model_info: Dict[str, Any] = {
        "key": catalog_model,
        "litellm_provider": "nvidia_nim",
        "mode": "rerank",
        "provider_specific_entry": {"rerank": 42},
    }
    with patch("litellm.utils.get_model_info", return_value=fake_model_info):
        with pytest.raises(TypeError, match="Malformed.*rerank"):
            get_nvidia_nim_model_metadata(catalog_model)


@pytest.mark.parametrize(
    "rerank_metadata",
    (
        {"endpoint_path": 123, "body_model": None},
        {"endpoint_path": "", "body_model": "nvidia/example-reranker"},
        {"endpoint_path": " \t", "body_model": "nvidia/example-reranker"},
        {"endpoint_path": "/v1/ranking", "body_model": ""},
        {"endpoint_path": "/v1/ranking", "body_model": " \n"},
    ),
)
@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_raise_typeerror_for_invalid_rerank_fields(
    rerank_metadata: Dict[str, Any],
) -> None:
    """Rerank endpoint_path/body_model must be non-empty strings."""
    catalog_model = next(iter(EXPECTED_NVIDIA_RERANK_CATALOG))
    fake_model_info: Dict[str, Any] = {
        "key": catalog_model,
        "litellm_provider": "nvidia_nim",
        "mode": "rerank",
        "provider_specific_entry": {"rerank": rerank_metadata},
    }
    with patch("litellm.utils.get_model_info", return_value=fake_model_info):
        with pytest.raises(
            TypeError,
            match="endpoint_path and body_model must be non-empty strings",
        ):
            get_nvidia_nim_model_metadata(catalog_model)


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_propagate_unexpected_resolver_errors() -> None:
    """Non-ValueError exceptions from get_model_info propagate."""
    catalog_model = next(iter(EXPECTED_NVIDIA_RERANK_CATALOG))
    with patch(
        "litellm.utils.get_model_info",
        side_effect=RuntimeError("unexpected resolver failure"),
    ):
        with pytest.raises(RuntimeError, match="unexpected resolver failure"):
            get_nvidia_nim_model_metadata(catalog_model)


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_not_swallow_errors_wrapped_by_real_model_info_resolver() -> None:
    """A confirmed catalog model must propagate failures wrapped by get_model_info."""
    catalog_model = next(iter(EXPECTED_NVIDIA_RERANK_CATALOG))
    with patch(
        "litellm.utils._get_model_info_from_model_cost",
        side_effect=RuntimeError("simulated catalog storage failure"),
    ):
        with pytest.raises(Exception, match="isn't mapped yet") as exc_info:
            get_nvidia_nim_model_metadata(catalog_model)

    assert isinstance(exc_info.value.__context__, RuntimeError)
    assert str(exc_info.value.__context__) == "simulated catalog storage failure"


@pytest.mark.usefixtures("local_nvidia_catalog")
def test_should_propagate_keyerror_from_resolver() -> None:
    """KeyError (not ValueError) from resolver propagates as-is."""
    catalog_model = next(iter(EXPECTED_NVIDIA_RERANK_CATALOG))
    with patch(
        "litellm.utils.get_model_info",
        side_effect=KeyError("missing_internal_key"),
    ):
        with pytest.raises(KeyError, match="missing_internal_key"):
            get_nvidia_nim_model_metadata(catalog_model)
