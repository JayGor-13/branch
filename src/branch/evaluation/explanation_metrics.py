"""Explanation Quality Score helpers."""

from __future__ import annotations

from typing import Any


def completeness(shap_result: dict[str, Any], narrative: str) -> float:
    features = [item["feature"] for item in shap_result.get("features", [])]
    if not features:
        return 0.0
    narrative_l = narrative.lower()
    mentioned = sum(1 for feature in features if feature.lower() in narrative_l)
    return mentioned / len(features)


def faithfulness(shap_result: dict[str, Any], narrative: str) -> float:
    features = shap_result.get("features", [])
    if not features:
        return 0.0
    narrative_l = narrative.lower()
    correct = 0
    cited = 0
    for item in features:
        if item["feature"].lower() not in narrative_l:
            continue
        cited += 1
        if item["direction"] == "increases_prediction" and "contributors toward" in narrative_l:
            correct += 1
        elif item["direction"] == "decreases_prediction" and "contributors away" in narrative_l:
            correct += 1
    return correct / cited if cited else 0.0


def clinical_alignment_score(guardrail_result: dict[str, Any]) -> float:
    status = guardrail_result.get("guardrail_status")
    if status == "no_anomaly":
        return 1.0
    if status in {"possible_anomaly", "insufficient_guideline_evidence"}:
        return 0.5
    return 0.0


def explanation_quality_score(
    faithfulness_score: float,
    completeness_score: float,
    alignment_score: float,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    return (
        alpha * faithfulness_score
        + beta * completeness_score
        + gamma * alignment_score
    )
