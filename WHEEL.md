# Callback Wheel

This fork ships the AAWM callback logic in two forms:

- Runtime source in `litellm/integrations/aawm_agent_identity.py`
- Published callback wheel source in `.wheel-build/`

Production uses the published wheel path for callback-only changes. The
production image pulls the wheel from the GitHub release artifact rather than
depending on LiteLLM repo code changes for parser-only updates.

## Source of truth

The wheel package lives under:

- `.wheel-build/pyproject.toml`
- `.wheel-build/aawm_litellm_callbacks/__init__.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`

The runtime callback and the wheel callback must stay behaviorally aligned.
Docstring/comment drift is acceptable; parsing and trace-name behavior is not.

## Current callback responsibilities

The callback wheel is no longer just a trace-name shim. It now owns the same
runtime behavior as the in-repo callback, including:

- Claude Code / agent identity extraction
- Langfuse trace naming and request-tag normalization
- Claude/Gemini reasoning and signature enrichment
- `public.session_history` persistence into the AAWM tristore
- Gemini/Codex usage breakout normalization for cache, reasoning, and tool-call fields

When the in-repo callback gains or changes one of these responsibilities, the
wheel source under `.wheel-build/` must be updated in the same change set.

## Release workflow

The callback wheel is released by:

- `.github/workflows/aawm-callback.yml`

That workflow:

1. builds the wheel from `.wheel-build/`
2. publishes it as a GitHub release on `cb-v*` tags
3. force-moves the `cb-latest` tag

## Production usage

Production consumes the published wheel from the GitHub release artifact. This
lets callback parsing changes ship independently from a full LiteLLM image
rebuild when the runtime integration surface does not otherwise change. The
current callback wheel line is intended to carry the full AAWM callback surface,
not only trace naming.

## Rebase rule

When rebasing this fork:

- keep `.wheel-build/` and `.github/workflows/aawm-callback.yml`
- verify the wheel callback and runtime callback still match
- treat wheel parity as part of the production acceptance bar
- bump `.wheel-build/pyproject.toml` when the wheel-visible behavior changes
