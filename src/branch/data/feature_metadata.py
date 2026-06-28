"""Feature constraints for BRANCH counterfactual generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from branch.utils.constants import (
    DATASET_MATERNAL_HEALTH,
    MATERNAL_FEATURES,
)
from branch.utils.io import write_json


MATERNAL_FEATURE_METADATA: dict[str, dict[str, Any]] = {
    "Age": {
        "type": "numeric",
        "unit": "years",
        "mutable": False,
        "clinical_actionability": "immutable",
        "permitted_range": [10, 70],
    },
    "Systolic BP": {
        "type": "numeric",
        "unit": "mmHg",
        "mutable": True,
        "clinical_actionability": "potentially_modifiable",
        "permitted_range": [70, 180],
    },
    "Diastolic": {
        "type": "numeric",
        "unit": "mmHg",
        "mutable": True,
        "clinical_actionability": "potentially_modifiable",
        "permitted_range": [40, 120],
    },
    "BS": {
        "type": "numeric",
        "unit": "mmol/L",
        "mutable": True,
        "clinical_actionability": "potentially_modifiable",
        "permitted_range": [4.0, 20.0],
    },
    "Body Temp": {
        "type": "numeric",
        "unit": "degF",
        "mutable": True,
        "clinical_actionability": "clinically_modifiable_signal",
        "permitted_range": [95.0, 105.0],
    },
    "BMI": {
        "type": "numeric",
        "unit": "kg/m2",
        "mutable": True,
        "clinical_actionability": "potentially_modifiable",
        "permitted_range": [12.0, 60.0],
    },
    "Previous Complications": {
        "type": "binary",
        "unit": "indicator",
        "mutable": False,
        "clinical_actionability": "historical_nonmodifiable",
        "permitted_range": [0, 1],
    },
    "Preexisting Diabetes": {
        "type": "binary",
        "unit": "indicator",
        "mutable": False,
        "clinical_actionability": "preexisting_condition",
        "permitted_range": [0, 1],
    },
    "Gestational Diabetes": {
        "type": "binary",
        "unit": "indicator",
        "mutable": True,
        "clinical_actionability": "clinically_modifiable_signal",
        "permitted_range": [0, 1],
    },
    "Mental Health": {
        "type": "binary",
        "unit": "indicator",
        "mutable": True,
        "clinical_actionability": "potentially_modifiable_support_need",
        "permitted_range": [0, 1],
    },
    "Heart Rate": {
        "type": "numeric",
        "unit": "beats_per_minute",
        "mutable": True,
        "clinical_actionability": "clinically_modifiable_signal",
        "permitted_range": [50, 130],
    },
}


def maternal_feature_metadata() -> dict[str, Any]:
    return {
        "dataset": DATASET_MATERNAL_HEALTH,
        "features": deepcopy(MATERNAL_FEATURE_METADATA),
        "feature_order": list(MATERNAL_FEATURES),
    }


def infer_feature_metadata(
    dataset: str,
    df,
    feature_names: list[str],
) -> dict[str, Any]:
    if dataset == DATASET_MATERNAL_HEALTH:
        return maternal_feature_metadata()

    metadata: dict[str, Any] = {
        "dataset": dataset,
        "features": {},
        "feature_order": list(feature_names),
    }
    for feature in feature_names:
        series = df[feature]
        low = float(series.min())
        high = float(series.max())
        if low == high:
            high = low + 1.0
        metadata["features"][feature] = {
            "type": "binary" if sorted(series.dropna().unique().tolist()) in [[0, 1], [0.0, 1.0]] else "numeric",
            "unit": "dataset_encoded",
            "mutable": _feature_is_mutable(dataset, feature),
            "clinical_actionability": _actionability_label(dataset, feature),
            "permitted_range": [_json_number(low), _json_number(high)],
        }
    return metadata


def _feature_is_mutable(dataset: str, feature: str) -> bool:
    lowered = feature.lower()
    immutable_terms = [
        "age",
        "gender",
        "sex",
        "race",
        "height",
        "comorbidity",
        "coronary artery disease",
        "hypothyroidism",
    ]
    if any(term in lowered for term in immutable_terms):
        return False
    if dataset == "gallstone" and "diabetes mellitus" in lowered:
        return False
    return True


def _actionability_label(dataset: str, feature: str) -> str:
    if not _feature_is_mutable(dataset, feature):
        return "immutable_or_historical"
    lowered = feature.lower()
    if any(term in lowered for term in ["cholesterol", "ldl", "hdl", "triglyceride", "glucose", "bmi", "weight", "blood", "bp"]):
        return "potentially_modifiable"
    if dataset == "npha":
        return "behavioral_or_survey_signal"
    return "clinically_modifiable_signal"


def mutable_features(metadata: dict[str, Any]) -> list[str]:
    features = metadata.get("features", metadata)
    return [name for name, spec in features.items() if spec.get("mutable", False)]


def immutable_features(metadata: dict[str, Any]) -> list[str]:
    features = metadata.get("features", metadata)
    return [name for name, spec in features.items() if not spec.get("mutable", False)]


def permitted_ranges(metadata: dict[str, Any]) -> dict[str, list[float]]:
    features = metadata.get("features", metadata)
    return {
        name: spec["permitted_range"]
        for name, spec in features.items()
        if "permitted_range" in spec
    }


def write_maternal_feature_metadata(path: str | Path) -> Path:
    return write_json(maternal_feature_metadata(), path)


def write_feature_metadata(
    dataset: str,
    df,
    feature_names: list[str],
    path: str | Path,
) -> Path:
    return write_json(infer_feature_metadata(dataset, df, feature_names), path)


def _json_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)
