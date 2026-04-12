from litellm.integrations.aawm_agent_identity import AawmAgentIdentity


def _base_kwargs(trace_name: str = "claude-code") -> dict:
    return {
        "litellm_params": {"metadata": {"trace_name": trace_name}},
        "standard_logging_object": {"metadata": {}, "request_tags": []},
        "passthrough_logging_payload": {
            "request_body": {
                "messages": [
                    {
                        "role": "user",
                        "content": "You are 'engineer' and you are working on the 'aegis' project.",
                    }
                ]
            }
        },
    }


def test_aawm_agent_identity_enriches_trace_name() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs()

    updated_kwargs, result = logger.logging_hook(
        kwargs=kwargs,
        result={"choices": []},
        call_type="pass_through_endpoint",
    )

    assert result == {"choices": []}
    assert (
        updated_kwargs["litellm_params"]["metadata"]["trace_name"]
        == "claude-code.engineer"
    )
    assert updated_kwargs["standard_logging_object"]["metadata"]["trace_name"] == (
        "claude-code.engineer"
    )


def test_aawm_agent_identity_adds_claude_thinking_tags() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs()
    result = {
        "choices": [
            {
                "message": {
                    "content": "Ready.",
                    "role": "assistant",
                    "provider_specific_fields": {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "EvMCCmwIDBgCKkAuqMZK8CbuNuz6LdJex7qr4ZB9T9OXQ6zOKvzTxK6SCYZUP3ageKCC1lr28wDIfpWyVJVPVGcFP+a5ScIJ6CsQMiJudW1iYXQtdjYtZWZmb3J0cy0yMC00MC04MC1hYi1wcm9kOAASDOBIjRsAoyR7Oe6UdhoMtmeDeP+RjpVIJjlYIjCq8O2tRhEP4C9HCx8SrqqE0v1cKQ6aiJBHzBOOdZZg92sNK/B/sL4ihlm+ptMA9NYqtAHTchkk3dthQhVBBayWWoOjP/IEZEhlzYHTdoTOzKkLUQNEeCmJQQst7E+ugv9gn+luB/SalmqboTM0FqmLYX8nWG5gMb8LI8ipTZwgLyYLoyvcg5NwaoWPqup1Wo4v85lJeoFam70xAyK7v2b1cDgNoYT+jVGRE4gUZy6W+ZOK7wxLdIkeObuEiAKjwKE6o8G6hfIB+AsW4mAOPymAOS8fm4JnYcz61kXO1MjvhtAqkjMNCPsYAQ==",
                            }
                        ]
                    },
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": " I'm ready and waiting for the user's question.",
                            "signature": "EvMCCmwIDBgCKkAuqMZK8CbuNuz6LdJex7qr4ZB9T9OXQ6zOKvzTxK6SCYZUP3ageKCC1lr28wDIfpWyVJVPVGcFP+a5ScIJ6CsQMiJudW1iYXQtdjYtZWZmb3J0cy0yMC00MC04MC1hYi1wcm9kOAASDOBIjRsAoyR7Oe6UdhoMtmeDeP+RjpVIJjlYIjCq8O2tRhEP4C9HCx8SrqqE0v1cKQ6aiJBHzBOOdZZg92sNK/B/sL4ihlm+ptMA9NYqtAHTchkk3dthQhVBBayWWoOjP/IEZEhlzYHTdoTOzKkLUQNEeCmJQQst7E+ugv9gn+luB/SalmqboTM0FqmLYX8nWG5gMb8LI8ipTZwgLyYLoyvcg5NwaoWPqup1Wo4v85lJeoFam70xAyK7v2b1cDgNoYT+jVGRE4gUZy6W+ZOK7wxLdIkeObuEiAKjwKE6o8G6hfIB+AsW4mAOPymAOS8fm4JnYcz61kXO1MjvhtAqkjMNCPsYAQ==",
                        }
                    ],
                    "reasoning_content": " I'm ready and waiting for the user's question.",
                }
            }
        ]
    }

    updated_kwargs, _ = logger.logging_hook(
        kwargs=kwargs,
        result=result,
        call_type="pass_through_endpoint",
    )

    metadata = updated_kwargs["litellm_params"]["metadata"]
    tags = metadata["tags"]
    assert metadata["trace_name"] == "claude-code.engineer"
    assert metadata["claude_thinking_signature_present"] is True
    assert metadata["claude_thinking_signature_count"] == 1
    assert metadata["claude_thinking_signature_decoded"] is True
    assert (
        metadata["claude_thinking_experiment_id"]
        == "numbat-v6-efforts-20-40-80-ab-prod"
    )
    assert metadata["claude_reasoning_content_present"] is True
    assert metadata["thinking_signature_present"] is True
    assert metadata["thinking_signature_decoded"] is True
    assert metadata["reasoning_content_present"] is True
    assert metadata["thinking_blocks_present"] is True
    assert "claude-thinking-signature" in tags
    assert "thinking-signature-present" in tags
    assert "thinking-signature-decoded" in tags
    assert "claude-thinking-decoded" in tags
    assert "claude-exp:numbat-v6-efforts-20-40-80-ab-prod" in tags
    assert "reasoning-present" in tags
    assert "thinking-blocks-present" in tags
    assert "claude-reasoning-present" in tags
    assert "claude-thinking-signature" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert "thinking-signature-present" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert (
        updated_kwargs["standard_logging_object"]["metadata"]["claude_thinking_signature_present"]
        is True
    )
    span_names = [
        span["name"] for span in metadata["langfuse_spans"] if isinstance(span, dict)
    ]
    assert "claude.thinking_signature_decode" in span_names
    claude_span = next(
        span
        for span in metadata["langfuse_spans"]
        if isinstance(span, dict)
        and span.get("name") == "claude.thinking_signature_decode"
    )
    assert claude_span["metadata"]["signature_count"] == 1
    assert claude_span["metadata"]["decoded_signature_count"] == 1
    assert claude_span["metadata"]["reasoning_content_present"] is True
    assert "start_time" in claude_span
    assert "end_time" in claude_span
    assert "gemini_thought_signature_present" not in metadata


def test_aawm_agent_identity_adds_gemini_thought_signature_tags() -> None:
    logger = AawmAgentIdentity()
    kwargs = _base_kwargs(trace_name="gemini")
    result = {
        "choices": [
            {
                "message": {
                    "content": "gemini routed",
                    "role": "assistant",
                    "thinking_blocks": [],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            """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"""
                        ]
                    },
                }
            }
        ]
    }

    updated_kwargs, _ = logger.logging_hook(
        kwargs=kwargs,
        result=result,
        call_type="pass_through_endpoint",
    )

    metadata = updated_kwargs["litellm_params"]["metadata"]
    tags = metadata["tags"]
    assert metadata["trace_name"] == "gemini"
    assert metadata["gemini_thought_signature_present"] is True
    assert metadata["gemini_thought_signature_count"] == 1
    assert metadata["gemini_tsig_record_count"] == 9
    assert metadata["gemini_tsig_record_sizes"] == [
        36,
        104,
        124,
        194,
        156,
        280,
        276,
        328,
        112,
    ]
    assert metadata["gemini_tsig_0_record_0_size"] == 36
    assert metadata["gemini_tsig_marker_hex"] == "8f3d6b5f"
    assert len(metadata["gemini_tsig_marker_offsets"]) == 9
    assert metadata["gemini_reasoning_content_present"] is False
    assert metadata["thinking_signature_present"] is True
    assert metadata["thinking_signature_decoded"] is True
    assert metadata["reasoning_content_present"] is False
    assert metadata["thinking_blocks_present"] is False
    assert "gemini-thought-signature" in tags
    assert "thinking-signature-present" in tags
    assert "thinking-signature-decoded" in tags
    assert "gemini-thought-signature-decoded" in tags
    assert "gemini-tsig-records:9" in tags
    assert "reasoning-empty" in tags
    assert "thinking-blocks-empty" in tags
    assert "gemini-reasoning-empty" in tags
    assert any(tag.startswith("gemini-tsig-shape:") for tag in tags)
    assert "gemini-thought-signature" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert "thinking-signature-present" in updated_kwargs["standard_logging_object"]["request_tags"]
    assert (
        updated_kwargs["standard_logging_object"]["metadata"]["gemini_thought_signature_present"]
        is True
    )
    span_names = [
        span["name"] for span in metadata["langfuse_spans"] if isinstance(span, dict)
    ]
    assert "gemini.thought_signature_decode" in span_names
    gemini_span = next(
        span
        for span in metadata["langfuse_spans"]
        if isinstance(span, dict)
        and span.get("name") == "gemini.thought_signature_decode"
    )
    assert gemini_span["metadata"]["signature_count"] == 1
    assert gemini_span["metadata"]["decoded_signature_count"] == 1
    assert gemini_span["metadata"]["record_counts"] == [9]
    assert "start_time" in gemini_span
    assert "end_time" in gemini_span
    assert "claude_thinking_signature_present" not in metadata
