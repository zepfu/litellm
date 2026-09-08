"""Typed records for the ChatGPT Chat usage collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class AdaptedPage:
    items: list[Any]
    continuation: str | int | None
    exhausted: bool
    schema_version: str
    coverage: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    is_archived: bool
    workspace_id: Optional[str]
    project_id: Optional[str]
    surface: str
    origin: Optional[str]
    has_versions: Optional[bool]
    current_node: Optional[str]
    coverage: str


@dataclass(frozen=True)
class MessageRecord:
    conversation_id: str
    message_id: str
    node_id: Optional[str]
    parent_id: Optional[str]
    children: tuple[str, ...]
    role: Optional[str]
    channel: Optional[str]
    created_at: Optional[datetime]
    status: Optional[str]
    end_turn: Optional[bool]
    requested_model_raw: Optional[str]
    requested_mode_raw: Optional[str]
    requested_reasoning_effort_raw: Optional[str]
    recorded_final_model_raw: Optional[str]
    generation_id: Optional[str]
    request_id: Optional[str]
    surface: str
    origin: Optional[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    conversation_id: str
    identity_basis: str
    time_basis: str
    attempt_time: Optional[datetime]
    earliest_possible_at: Optional[datetime]
    latest_possible_at: Optional[datetime]
    requested_model_raw: Optional[str]
    requested_mode_raw: Optional[str]
    requested_reasoning_effort_raw: Optional[str]
    recorded_final_model_raw: Optional[str]
    resolved_model_raw: Optional[str]
    requested_family: Optional[str]
    recorded_final_family: Optional[str]
    resolved_family: Optional[str]
    mapping_version: str
    outcome: str
    completed_answer: bool
    generation_started: bool
    surface: str
    origin: Optional[str]
    aliases: tuple[tuple[str, str], ...]
    evidence_message_ids: tuple[str, ...]
    revision: int
    warnings: tuple[str, ...]
    quarantine: Optional[Mapping[str, Any]] = None
    source_identity_basis: Optional[str] = None
    source_time_basis: Optional[str] = None
    source_outcome: Optional[str] = None


@dataclass(frozen=True)
class AccountBinding:
    collector_account_id: str
    provider_user_id: Optional[str]
    workspace_id: Optional[str]
    quota_owner_id: str
    surface: str
    auth_state: str
    plan_policy_id: str
    enabled: bool


@dataclass(frozen=True)
class CapabilityRecord:
    adapter_version: str
    index_scopes: tuple[str, ...]
    archive_behavior: str
    project_coverage: str
    modern_detail: str
    pagination: str
    legacy_support: str
    branch_visibility: str
    model_metadata: str
    quota_metadata: str
    warnings: tuple[str, ...] = ()
