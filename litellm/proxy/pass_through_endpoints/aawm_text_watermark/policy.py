"""Detect / sanitize / enforce policy machine and bounded audit builders."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .config import (
    OpenAIPassthroughTextWatermarkSettings,
    load_text_watermark_config,
)
from .text_nodes import (
    VisibleTextNode,
    assign_text_node,
    extract_visible_text_nodes,
)
from .unicode_detector import (
    UNICODE_CARRIER_DETECTOR_NAME,
    UNICODE_CARRIER_DETECTOR_VERSION,
    UnicodeCarrierDetection,
    detect_unicode_carriers,
    sanitize_unicode_carriers,
)

AUDIT_SCHEMA_VERSION = 1


@dataclass
class WatermarkPolicyResult:
    body: dict[str, Any]
    audit: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class StatisticalDetectorEvaluation:
    status: str
    detectors: tuple[dict[str, Any], ...] = ()


def _coerce_config(config: Any) -> OpenAIPassthroughTextWatermarkSettings:
    if isinstance(config, OpenAIPassthroughTextWatermarkSettings):
        return config
    return load_text_watermark_config(config)


def _direction_enabled(config: OpenAIPassthroughTextWatermarkSettings, direction: str) -> bool:
    directions = config.directions
    if direction == "response":
        return bool(directions.response)
    return bool(directions.request)


def _collect_nodes(
    body: Mapping[str, Any],
    *,
    endpoint: str,
    direction: str,
    config: OpenAIPassthroughTextWatermarkSettings,
) -> tuple[list[VisibleTextNode], int, int, bool]:
    limits = config.limits
    scanned: list[VisibleTextNode] = []
    skipped = 0
    scanned_bytes = 0
    truncated = False
    for node in extract_visible_text_nodes(
        body, endpoint=endpoint, direction=direction
    ):
        encoded = len(node.text.encode("utf-8"))
        if len(scanned) >= limits.max_text_nodes_per_direction:
            skipped += 1
            truncated = True
            continue
        if scanned_bytes + encoded > limits.max_text_bytes_per_direction:
            skipped += 1
            truncated = True
            continue
        scanned.append(node)
        scanned_bytes += encoded
    return scanned, skipped, scanned_bytes, truncated


def _path_audit(
    node: VisibleTextNode,
    detection: UnicodeCarrierDetection,
    max_hits: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": node.path,
        "role": node.role,
        "hit_count": detection.hit_count,
        "hit_kinds": list(detection.hit_kinds),
    }
    if detection.hits and max_hits > 0:
        capped = detection.hits[:max_hits]
        payload["hits"] = [
            {"kind": hit.kind, "index": hit.index} for hit in capped
        ]
    return payload


def _merge_kinds(kind_lists: Sequence[Sequence[str]]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for kinds in kind_lists:
        for kind in kinds:
            if kind in seen:
                continue
            seen.add(kind)
            merged.append(kind)
    return merged


def _unicode_detector_audit(
    *,
    hit_count: int,
    hit_kinds: Sequence[str],
    status: str,
) -> dict[str, Any]:
    return {
        "name": UNICODE_CARRIER_DETECTOR_NAME,
        "version": UNICODE_CARRIER_DETECTOR_VERSION,
        "status": status,
        "confidence": "probable" if hit_count else "none",
        "hit_count": hit_count,
        "hit_kinds": list(hit_kinds),
    }


def _build_audit(
    *,
    direction: str,
    mode: str,
    status: str,
    signal_detected: bool,
    scanned_text_nodes: int,
    scanned_text_bytes: int,
    skipped_text_nodes: int,
    detectors: list[dict[str, Any]],
    transformation: dict[str, Any],
    paths: list[dict[str, Any]],
    truncated: bool,
    errors: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "direction": direction,
        "mode": mode,
        "status": status,
        "signal_detected": signal_detected,
        "confirmed_watermark_detected": False,
        "vendor_attribution": "unknown",
        "scanned_text_nodes": scanned_text_nodes,
        "scanned_text_bytes": scanned_text_bytes,
        "skipped_text_nodes": skipped_text_nodes,
        "detectors": detectors,
        "transformation": transformation,
        "paths": paths,
        "truncated": truncated,
        "errors": list(errors or []),
    }


def evaluate_statistical_detectors(
    text: str,
    config: Any = None,
    **kwargs: Any,
) -> StatisticalDetectorEvaluation:
    """Empty/disabled registry is unsupported. Never imports torch."""

    del text, kwargs
    loaded = _coerce_config(config)
    summaries: list[dict[str, Any]] = []
    for detector in loaded.statistical_detectors:
        summaries.append(
            {
                "name": detector.name,
                "type": detector.type,
                "enabled": detector.enabled,
                "status": "unsupported",
            }
        )
    return StatisticalDetectorEvaluation(
        status="unsupported",
        detectors=tuple(summaries),
    )


def apply_watermark_policy(
    body: dict[str, Any],
    config: Any,
    direction: str = "request",
    endpoint: str = "responses",
    **kwargs: Any,
) -> WatermarkPolicyResult:
    """Run detect, optional sanitize, re-detect. ``mode=off`` is a no-op."""

    del kwargs
    if not isinstance(body, dict):
        return WatermarkPolicyResult(body=body, audit=None)
    loaded = _coerce_config(config)
    if loaded.mode == "off" or not _direction_enabled(loaded, direction):
        return WatermarkPolicyResult(body=body, audit=None)

    nodes, skipped, scanned_bytes, truncated_limits = _collect_nodes(
        body, endpoint=endpoint, direction=direction, config=loaded
    )
    unicode_settings = loaded.unicode
    policy_name = unicode_settings.policy
    mutate = (
        loaded.mode in {"sanitize", "enforce"}
        and loaded.removal.enabled
        and unicode_settings.enabled
    )
    out_body = copy.deepcopy(body) if mutate else body

    pre_hits = 0
    pre_kinds: list[list[str]] = []
    path_rows: list[dict[str, Any]] = []
    removed_total = 0
    replaced_total = 0
    post_hits = 0
    errors: list[str] = []

    for node in nodes:
        try:
            detection = (
                detect_unicode_carriers(
                    node.text,
                    policy=policy_name,
                    normalize_spaces=unicode_settings.normalize_spaces,
                )
                if unicode_settings.enabled
                else UnicodeCarrierDetection(signal_detected=False)
            )
        except Exception as exc:  # fail-soft in detect/sanitize
            errors.append(type(exc).__name__)
            detection = UnicodeCarrierDetection(signal_detected=False)
        if detection.hit_count:
            pre_hits += detection.hit_count
            pre_kinds.append(list(detection.hit_kinds))
            path_rows.append(
                _path_audit(
                    node, detection, loaded.limits.max_reported_hits_per_path
                )
            )
        if not mutate:
            continue
        sanitized = sanitize_unicode_carriers(
            node.text,
            policy=policy_name,
            normalize_spaces=unicode_settings.normalize_spaces,
            nfkc=unicode_settings.nfkc,
        )
        removed_total += sanitized.removed_count
        replaced_total += sanitized.replaced_count
        if sanitized.text != node.text:
            assign_text_node(out_body, node, sanitized.text)
        post = detect_unicode_carriers(
            sanitized.text,
            policy=policy_name,
            normalize_spaces=unicode_settings.normalize_spaces,
        )
        post_hits += post.hit_count

    merged_pre_kinds = _merge_kinds(pre_kinds)
    max_paths = loaded.limits.max_reported_paths
    truncated = truncated_limits or len(path_rows) > max_paths
    reported_paths = path_rows[:max_paths]

    signal_detected = pre_hits > 0
    if mutate:
        transformation_result = "not_requested"
        if not signal_detected and removed_total == 0 and replaced_total == 0:
            transformation_result = "not_requested"
            post_status = "clean"
            status = "clean"
        elif post_hits == 0:
            transformation_result = "removed_verified"
            post_status = "removed_verified"
            status = "sanitized"
        elif removed_total or replaced_total:
            transformation_result = "partially_sanitized"
            post_status = "signal_remaining"
            status = "partially_sanitized"
        else:
            transformation_result = "detected_unremoved"
            post_status = "signal_remaining"
            status = "detected"
        if loaded.mode == "enforce" and post_hits > 0:
            status = "blocked"
        transformation = {
            "attempted": True,
            "policy": policy_name,
            "removed_count": removed_total,
            "replaced_count": replaced_total,
            "post_status": post_status,
            "result": transformation_result,
        }
        detector_status = "detected" if pre_hits else "clean"
        detector_kinds = merged_pre_kinds
        detector_hits = pre_hits
    else:
        status = "detected" if signal_detected else "clean"
        transformation = {
            "attempted": False,
            "policy": policy_name,
            "removed_count": 0,
            "replaced_count": 0,
            "post_status": "not_requested",
            "result": "not_requested",
        }
        detector_status = "detected" if signal_detected else "clean"
        detector_kinds = merged_pre_kinds
        detector_hits = pre_hits

    if errors and loaded.mode != "enforce":
        status = "error"

    statistical = evaluate_statistical_detectors("", config=loaded)
    detectors = []
    if unicode_settings.enabled:
        detectors.append(
            _unicode_detector_audit(
                hit_count=detector_hits,
                hit_kinds=detector_kinds,
                status=detector_status,
            )
        )
    if statistical.detectors:
        detectors.extend(list(statistical.detectors))
    elif not loaded.statistical_detectors:
        detectors.append(
            {
                "name": "statistical_registry",
                "version": 1,
                "status": statistical.status,
                "confidence": "none",
                "hit_count": 0,
                "hit_kinds": [],
            }
        )

    audit = _build_audit(
        direction=direction,
        mode=loaded.mode,
        status=status,
        signal_detected=signal_detected,
        scanned_text_nodes=len(nodes),
        scanned_text_bytes=scanned_bytes,
        skipped_text_nodes=skipped,
        detectors=detectors,
        transformation=transformation,
        paths=reported_paths,
        truncated=truncated,
        errors=errors,
    )
    return WatermarkPolicyResult(body=out_body, audit=audit)
