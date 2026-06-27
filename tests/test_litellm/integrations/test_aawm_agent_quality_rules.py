from litellm.integrations.aawm_agent_quality_rules import (
    AgentQualityCommand,
    score_agent_quality_context,
)


def test_score_agent_quality_context_returns_core_policy_fields() -> None:
    result = score_agent_quality_context(
        user_texts=["hello"],
        assistant_texts=["world"],
        tool_result_texts=["some tool output"],
        commands=[AgentQualityCommand(name="git", command="git status")],
    )

    assert "ignored_path_tracking_policy_score" in result.fields
    assert "discovery_inventory_coverage_score" in result.fields


def test_score_agent_quality_context_flags_literal_composer_call_text() -> None:
    result = score_agent_quality_context(
        assistant_texts=['{"name": "composer_call", "arguments": {"cmd": "pwd"}}'],
    )

    assert result.fields["output_contract_compliance_score"] == 0.0
    assert result.fields["output_contract_failure_class"] == "literal_tool_call_text"
    assert result.reasons["output_contract_compliance"] == [
        "literal_tool_call_text"
    ]


def test_score_agent_quality_context_flags_composer_call_tool_marker() -> None:
    result = score_agent_quality_context(
        assistant_texts=["Done."],
        tool_call_names=["composer_call"],
    )

    assert result.fields["output_contract_compliance_score"] == 0.0
    assert result.fields["output_contract_failure_class"] == "malformed_tool_call_text"
    assert result.reasons["output_contract_compliance"] == [
        "malformed_tool_call_text"
    ]


def test_score_agent_quality_context_ignores_benign_composer_call_prose() -> None:
    result = score_agent_quality_context(
        user_texts=["Explain what happened with the engineer route."],
        assistant_texts=[
            "The model discussed composer_call in prose and quoted "
            "call-123-composer_call_abc without emitting a tool call."
        ],
    )

    assert result.fields.get("output_contract_failure_class") is None
    assert result.reasons.get("output_contract_compliance") == []


def test_score_agent_quality_context_flags_same_line_serialized_composer_call_text() -> None:
    result = score_agent_quality_context(
        assistant_texts=[
            'Name: Bash  Call ID: call-abc-composer_call_qz904 Arguments: {"command":"grep"}'
        ],
    )

    assert result.fields["output_contract_compliance_score"] == 0.0
    assert result.fields["output_contract_failure_class"] == "literal_tool_call_text"
    assert result.reasons["output_contract_compliance"] == [
        "literal_tool_call_text"
    ]


def test_score_agent_quality_context_flags_literal_claude_xml_invoke_text() -> None:
    result = score_agent_quality_context(
        assistant_texts=['<invoke name="Bash">\n<parameter name="command">pwd</parameter>\n</invoke>'],
    )

    assert result.fields["output_contract_compliance_score"] == 0.0
    assert result.fields["output_contract_failure_class"] == "literal_tool_call_text"
    assert result.reasons["output_contract_compliance"] == [
        "literal_tool_call_text"
    ]


def test_score_agent_quality_context_ignores_benign_claude_xml_prose() -> None:
    result = score_agent_quality_context(
        user_texts=["Explain the failure mode."],
        assistant_texts=[
            'The model printed the string <invoke name="Bash"> in prose but did not emit tool_use.'
        ],
    )

    assert result.fields.get("output_contract_failure_class") is None
    assert result.reasons.get("output_contract_compliance") == []


def test_score_agent_quality_context_does_not_flag_structured_tool_use_as_literal_xml() -> None:
    result = score_agent_quality_context(
        assistant_texts=["Done."],
        tool_call_names=["Bash"],
    )

    assert result.fields.get("output_contract_failure_class") is None
    assert result.reasons.get("output_contract_compliance") == []


def test_score_agent_quality_context_ignores_request_side_tool_use_error_text() -> None:
    result = score_agent_quality_context(
        user_texts=[
            (
                "<tool_use_error>Previous assistant text contained "
                '<invoke name="Bash"> but the current answer should not inherit it.'
                "</tool_use_error>"
            )
        ],
        assistant_texts=["Done."],
    )

    assert result.fields.get("output_contract_failure_class") is None
    assert result.reasons.get("output_contract_compliance") == []


def test_ignored_path_tracking_flags_forced_adds() -> None:
    result = score_agent_quality_context(
        commands=[
            AgentQualityCommand(
                name="git",
                command="git add -f .analysis/todo.md",
            )
        ]
    )

    assert result.fields["ignored_path_tracking_policy_score"] == 0.0
    assert result.fields["ignored_path_tracking_violation_count"] == 1
    assert result.fields["ignored_path_tracking_evidence"][0]["command_snippet"] == (
        "git add -f .analysis/todo.md"
    )


def test_discovery_inventory_required_task_flags_missing_inventory() -> None:
    result = score_agent_quality_context(
        user_texts=[
            "Discovery inventory required: list the discovery command, "
            "list every candidate, and classify relevant candidates for this "
            "contract review."
        ],
        assistant_texts=["I checked the files and found nothing."],
        tool_result_texts=["src/example.py"],
    )

    assert result.fields["discovery_inventory_coverage_score"] == 0.0
    assert result.fields["discovery_inventory_missing_count"] > 0
    reasons = {
        item["reason"] for item in result.fields["discovery_inventory_evidence"]
    }
    assert "missing_inventory_section" in reasons


def test_discovery_inventory_required_task_accepts_complete_inventory() -> None:
    result = score_agent_quality_context(
        user_texts=[
            "Discovery inventory required: list the discovery command, "
            "list every candidate, mark each candidate as inspected, classify "
            "relevant candidates, and call out any coverage gap for this "
            "contract review."
        ],
        assistant_texts=[
            "Discovery command: rg --files. Candidate inventory: "
            "src/example.py inspected and actionable. Coverage gap: none."
        ],
        tool_result_texts=["src/example.py"],
        commands=[
            AgentQualityCommand(
                name="rg",
                command="rg --files",
                affected_paths=("src/example.py",),
            )
        ],
    )

    assert result.fields["discovery_inventory_coverage_score"] == 1.0
    assert result.fields["discovery_inventory_missing_count"] == 0
    assert result.fields["discovery_inventory_evidence"] == []
