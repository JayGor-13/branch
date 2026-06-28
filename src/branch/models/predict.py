"""Structured prediction output for BRANCH tools."""

from __future__ import annotations

from typing import Any, Mapping

from branch.utils.constants import MATERNAL_ID_TO_LABEL, PATIENT_ID
from branch.utils.dependencies import require


def _patient_frame(patient: Mapping[str, Any], feature_names: list[str]):
    pd = require("pandas")
    return pd.DataFrame([{feature: patient[feature] for feature in feature_names}])


def predict_patient(
    model: Any,
    patient: Mapping[str, Any],
    feature_names: list[str],
    label_order: list[str],
    dataset: str = "maternal_health",
    task_type: str = "multiclass_classification",
) -> dict[str, Any]:
    np = require("numpy")

    frame = _patient_frame(patient, feature_names)
    patient_id = patient.get(PATIENT_ID)

    if task_type == "regression":
        value = float(np.asarray(model.predict(frame))[0])
        return {
            "dataset": dataset,
            "patient_id": None if patient_id is None else int(patient_id),
            "task_type": "regression",
            "predicted_value": value,
            "prediction": value,
            "predicted_class_id": None,
            "predicted_class": "continuous_prediction",
            "predicted_probability": None,
            "class_probabilities": {},
        }

    probabilities = model.predict_proba(frame)[0]
    predicted_id = int(np.argmax(probabilities))

    return {
        "dataset": dataset,
        "patient_id": None if patient_id is None else int(patient_id),
        "task_type": task_type,
        "predicted_class_id": predicted_id,
        "predicted_class": label_order[predicted_id]
        if predicted_id < len(label_order)
        else MATERNAL_ID_TO_LABEL.get(predicted_id, str(predicted_id)),
        "predicted_probability": float(probabilities[predicted_id]),
        "class_probabilities": {
            label_order[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
            if idx < len(label_order)
        },
    }
