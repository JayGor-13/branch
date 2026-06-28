"""RAG-grounded clinical alignment checks for BRANCH explanations."""

from __future__ import annotations

from typing import Any


def model_risk_direction(shap_value: float, predicted_class: str) -> str:
    predicted_l = str(predicted_class).lower()
    if any(term in predicted_l for term in ["high", "present", "positive"]):
        return "increases_risk" if shap_value >= 0 else "decreases_risk"
    if any(term in predicted_l for term in ["low", "not", "absent", "negative", "no "]):
        return "decreases_risk" if shap_value >= 0 else "increases_risk"
    if predicted_l == "continuous_prediction":
        return "increases_risk" if shap_value >= 0 else "decreases_risk"
    return "unclear_for_mid_risk"


def clinical_direction(feature: str, value: float, dataset: str | None = None) -> str:
    value = float(value)
    feature_l = feature.lower()
    dataset = dataset or ""
    if feature == "Systolic BP":
        return "increases_risk" if value >= 140 else "insufficient_evidence"
    if feature == "Diastolic":
        return "increases_risk" if value >= 90 else "insufficient_evidence"
    if feature == "BS":
        return "increases_risk" if value >= 7.8 else "insufficient_evidence"
    if feature == "Body Temp":
        return "increases_risk" if value >= 100.4 else "insufficient_evidence"
    if feature == "BMI":
        return "increases_risk" if value >= 30 else "insufficient_evidence"
    if feature in {
        "Previous Complications",
        "Preexisting Diabetes",
        "Gestational Diabetes",
        "Mental Health",
    }:
        return "increases_risk" if value >= 1 else "insufficient_evidence"
    if feature == "Heart Rate":
        return "increases_risk" if value >= 100 else "insufficient_evidence"
    if feature == "Age":
        return "increases_risk" if value <= 18 or value >= 35 else "insufficient_evidence"
    if dataset == "gallstone":
        if any(term in feature_l for term in ["cholesterol", "ldl", "triglyceride", "glucose", "bmi", "fat", "alt", "ast"]):
            return "increases_risk"
        if "hdl" in feature_l:
            return "decreases_risk"
        if feature == "Age":
            return "increases_risk" if value >= 50 else "insufficient_evidence"
    if dataset == "npha":
        if any(term in feature_l for term in ["trouble sleeping", "pain", "medication", "mental health", "physical health", "dental health"]):
            return "increases_risk"
    if dataset == "load_diabetes":
        if feature_l in {"bmi", "bp", "s5", "s6"}:
            return "increases_risk"
    return "insufficient_evidence"


def clinical_direction_from_guidelines(
    feature: str,
    value: float,
    retrieved_chunks: list[dict[str, Any]],
    dataset: str | None = None,
) -> tuple[str, list[str]]:
    evidence_chunks = []
    retrieved_direction = None
    for chunk in retrieved_chunks:
        directions = chunk.get("feature_directions", {})
        if feature in directions:
            retrieved_direction = directions[feature]
            evidence_chunks.append(chunk.get("chunk_id", "unknown_chunk"))

    if not evidence_chunks:
        return "insufficient_evidence", []

    value = float(value)
    if retrieved_direction == "presence_increases_risk":
        return (
            "increases_risk" if value >= 1 else "insufficient_evidence",
            evidence_chunks,
        )
    if retrieved_direction == "high_value_increases_risk":
        direction = clinical_direction(feature, value, dataset=dataset)
        return direction, evidence_chunks
    if retrieved_direction == "low_value_increases_risk":
        return (
            "increases_risk" if float(value) <= 0 else "insufficient_evidence",
            evidence_chunks,
        )
    if retrieved_direction == "extreme_value_increases_risk":
        direction = clinical_direction(feature, value, dataset=dataset)
        return direction, evidence_chunks
    if retrieved_direction == "poor_status_increases_risk":
        return "increases_risk", evidence_chunks
    if retrieved_direction == "protective":
        return "decreases_risk", evidence_chunks

    return "insufficient_evidence", evidence_chunks


def compare_alignment(model_direction: str, guideline_direction: str) -> str:
    if "unclear" in model_direction or guideline_direction == "insufficient_evidence":
        return "unclear"
    if model_direction == guideline_direction:
        return "concordant"
    return "discordant"


def check_clinical_alignment(
    shap_result: dict[str, Any],
    guideline_context: dict[str, Any],
) -> dict[str, Any]:
    retrieved = guideline_context.get("retrieved_chunks", [])
    if not retrieved:
        return {
            "dataset": shap_result.get("dataset"),
            "patient_id": shap_result.get("patient_id"),
            "guardrail_status": "retrieval_failed",
            "retrieved_chunks": [],
            "alignment_checks": [],
            "warning": "No guideline chunks were retrieved for the explanation.",
        }

    predicted_class = shap_result.get("predicted_class", "")
    dataset = shap_result.get("dataset")
    checks = []
    for item in shap_result.get("features", []):
        feature = item["feature"]
        model_direction = model_risk_direction(float(item["shap"]), predicted_class)
        guideline_direction, evidence_chunk_ids = clinical_direction_from_guidelines(
            feature, item["value"], retrieved, dataset=dataset
        )
        checks.append(
            {
                "feature": feature,
                "value": item["value"],
                "model_direction": model_direction,
                "guideline_direction": guideline_direction,
                "alignment": compare_alignment(model_direction, guideline_direction),
                "evidence_source": "retrieved_guideline_chunks"
                if evidence_chunk_ids
                else "no_retrieved_guideline_evidence",
                "evidence_chunk_ids": evidence_chunk_ids,
            }
        )

    status = guardrail_status_from_checks(checks)
    result = {
        "dataset": shap_result.get("dataset"),
        "patient_id": shap_result.get("patient_id"),
        "guardrail_status": status,
        "retrieved_chunks": retrieved,
        "alignment_checks": checks,
    }
    if status == "anomaly_detected":
        result["warning"] = (
            "At least one model feature direction conflicts with the curated "
            "clinical direction rule. Treat this explanation with caution."
        )
    elif status == "insufficient_guideline_evidence":
        result["warning"] = (
            "Retrieved guidance did not provide enough directional evidence for "
            "the top model drivers."
        )
    return result


def guardrail_status_from_checks(checks: list[dict[str, Any]]) -> str:
    if any(item["alignment"] == "discordant" for item in checks):
        return "anomaly_detected"
    if any(item["alignment"] == "concordant" for item in checks) and any(
        item["alignment"] == "unclear" for item in checks
    ):
        return "possible_anomaly"
    if checks and all(item["alignment"] == "concordant" for item in checks):
        return "no_anomaly"
    return "insufficient_guideline_evidence"
