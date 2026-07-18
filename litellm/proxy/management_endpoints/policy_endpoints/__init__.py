"""
Policy endpoints package.

Re-exports everything from endpoints module so existing imports
like `from litellm.proxy.management_endpoints.policy_endpoints import router`
continue to work. Patch targets also resolve correctly since names
are imported directly into this namespace.
"""

import sys

from litellm.proxy.management_endpoints.policy_endpoints import (
    endpoints as _endpoints,
)

router = _endpoints.router
apply_policies = _endpoints.apply_policies
_build_all_names_per_competitor = _endpoints._build_all_names_per_competitor
_build_comparison_blocked_words = _endpoints._build_comparison_blocked_words
_build_competitor_guardrail_definitions = (
    _endpoints._build_competitor_guardrail_definitions
)
_build_name_blocked_words = _endpoints._build_name_blocked_words
_build_recommendation_blocked_words = (
    _endpoints._build_recommendation_blocked_words
)
_build_refinement_prompt = _endpoints._build_refinement_prompt
_clean_competitor_line = _endpoints._clean_competitor_line
_parse_variations_response = _endpoints._parse_variations_response

_PUBLIC_NAMES = tuple(
    name for name in vars(_endpoints) if not name.startswith("_")
)
_PACKAGE_MODULE = sys.modules[__name__]
for _name in _PUBLIC_NAMES:
    setattr(_PACKAGE_MODULE, _name, getattr(_endpoints, _name))

__all__ = [*_PUBLIC_NAMES]
