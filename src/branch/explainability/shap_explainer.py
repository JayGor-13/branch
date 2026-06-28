"""SHAP local explanations for the XGBoost maternal risk predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from branch.explainability.explanation_schema import validate_shap_result
from branch.models.predict import predict_patient
from branch.utils.constants import PATIENT_ID
from branch.utils.dependencies import require
from branch.utils.io import write_json


def _patient_frame(patient: Mapping[str, Any], feature_names: list[str]):
    pd = require("pandas")
    return pd.DataFrame([{feature: patient[feature] for feature in feature_names}])


def _select_class_values(raw_values, class_idx: int):
    np = require("numpy")

    if isinstance(raw_values, list):
        return np.asarray(raw_values[class_idx])[0]

    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[0, :, class_idx]
    if values.ndim == 2:
        return values[0]
    raise ValueError(f"Unexpected SHAP value shape: {values.shape}")


def _select_base_value(expected_value, class_idx: int) -> float:
    np = require("numpy")

    if isinstance(expected_value, list):
        return float(expected_value[class_idx])
    values = np.asarray(expected_value)
    if values.ndim == 0:
        return float(values)
    return float(values[class_idx])


def explain_patient_shap(
    model: Any,
    patient: Mapping[str, Any],
    feature_names: list[str],
    label_order: list[str],
    dataset: str = "maternal_health",
    task_type: str = "multiclass_classification",
    top_k: int = 5,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    shap = require("shap")
    np = require("numpy")

    frame = _patient_frame(patient, feature_names)
    prediction = predict_patient(
        model,
        patient,
        feature_names,
        label_order,
        dataset,
        task_type=task_type,
    )
    class_idx = int(prediction["predicted_class_id"] or 0)

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(frame)
    if task_type == "regression":
        class_values = _select_regression_values(raw_values)
        base_value = _select_base_value(explainer.expected_value, 0)
    else:
        class_values = _select_class_values(raw_values, class_idx)
        base_value = _select_base_value(explainer.expected_value, class_idx)

    ranked = np.argsort(np.abs(class_values))[::-1][:top_k]
    features = []
    for rank, idx in enumerate(ranked, start=1):
        shap_value = float(class_values[idx])
        features.append(
            {
                "rank": rank,
                "feature": feature_names[idx],
                "value": _json_scalar(patient[feature_names[idx]]),
                "shap": shap_value,
                "direction": "increases_prediction"
                if shap_value >= 0
                else "decreases_prediction",
            }
        )

    result = {
        "dataset": dataset,
        "patient_id": int(patient[PATIENT_ID]) if PATIENT_ID in patient else None,
        "model": "xgboost",
        "base_value": base_value,
        "prediction": prediction,
        "predicted_class": prediction["predicted_class"],
        "predicted_probability": prediction["predicted_probability"],
        "top_k": int(top_k),
        "features": features,
    }
    validate_shap_result(result)
    if output_path:
        write_json(result, output_path)
    return result


def _json_scalar(value):
    try:
        return value.item()
    except AttributeError:
        return value


def _select_regression_values(raw_values):
    np = require("numpy")

    values = np.asarray(raw_values)
    if values.ndim == 2:
        return values[0]
    if values.ndim == 1:
        return values
    raise ValueError(f"Unexpected regression SHAP value shape: {values.shape}")
