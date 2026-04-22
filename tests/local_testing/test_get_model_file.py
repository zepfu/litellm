import os, sys, traceback
import importlib.resources
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
import pytest


def test_get_model_cost_map():
    try:
        print(litellm.get_model_cost_map(url="fake-url"))
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_get_bundled_fallback_model_cost_map():
    with importlib.resources.open_text(
        "litellm", "bundled_model_prices_and_context_window_fallback.json"
    ) as f:
        print("inside bundled fallback")
        content = json.load(f)
        print("content", content)
