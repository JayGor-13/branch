"""Lightweight schema checks for saved explanation artifacts."""

from __future__ import annotations


REQUIRED_SHAP_KEYS = {
    "dataset",
    "patient_id",
    "model",
    "base_value",
    "prediction",
    "top_k",
    "features",
}

REQUIRED_DICE_KEYS = {
    "dataset",
    "patient_id",
    "original_prediction",
    "counterfactual_prediction",
    "distance",
    "changes",
    "validity",
}


def validate_shap_result(result: dict) -> None:
    missing = REQUIRED_SHAP_KEYS - set(result)
    if missing:
        raise ValueError(f"SHAP result missing keys: {sorted(missing)}")
    for item in result["features"]:
        for key in ["rank", "feature", "value", "shap", "direction"]:
            if key not in item:
                raise ValueError(f"SHAP feature entry missing key '{key}'")


def validate_dice_result(result: dict) -> None:
    missing = REQUIRED_DICE_KEYS - set(result)
    if missing:
        raise ValueError(f"DiCE result missing keys: {sorted(missing)}")
    for item in result["changes"]:
        for key in ["feature", "from", "to", "mutable"]:
            if key not in item:
                raise ValueError(f"DiCE change entry missing key '{key}'")
