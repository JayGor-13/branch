"""Anomaly status helpers."""

from __future__ import annotations

from typing import Any

from branch.guardrails.alignment_checker import guardrail_status_from_checks


def detect_anomaly(alignment_result: dict[str, Any]) -> dict[str, Any]:
    checks = alignment_result.get("alignment_checks", [])
    status = guardrail_status_from_checks(checks)
    enriched = dict(alignment_result)
    enriched["guardrail_status"] = status
    enriched["anomaly_detected"] = status == "anomaly_detected"
    return enriched
