"""DiCE counterfactual generation with maternal feature constraints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from branch.data.dataset_specs import get_dataset_spec
from branch.data.feature_metadata import mutable_features, permitted_ranges
from branch.explainability.explanation_schema import validate_dice_result
from branch.models.predict import predict_patient
from branch.utils.constants import PATIENT_ID
from branch.utils.dependencies import require
from branch.utils.io import write_json


def desired_lower_risk_class(predicted_class_id: int) -> int:
    if predicted_class_id <= 0:
        return 0
    return predicted_class_id - 1


def normalized_l1_distance(
    original: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    ranges: dict[str, list[float]],
) -> float:
    total = 0.0
    count = 0
    for feature, bounds in ranges.items():
        low, high = bounds
        width = max(float(high) - float(low), 1e-9)
        total += abs(float(counterfactual[feature]) - float(original[feature])) / width
        count += 1
    return float(total / max(count, 1))


def validate_counterfactual_changes(
    original: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    feature_metadata: dict[str, Any],
) -> tuple[str, list[str]]:
    features = feature_metadata.get("features", feature_metadata)
    problems: list[str] = []
    for feature, spec in features.items():
        original_value = float(original[feature])
        cf_value = float(counterfactual[feature])
        if not spec.get("mutable", False) and abs(cf_value - original_value) > 1e-8:
            problems.append(f"Immutable feature changed: {feature}")
        if "permitted_range" in spec:
            low, high = spec["permitted_range"]
            if cf_value < float(low) or cf_value > float(high):
                problems.append(f"Feature outside permitted range: {feature}")
    return ("valid" if not problems else "invalid", problems)


def generate_dice_counterfactual(
    model: Any,
    train_df,
    patient: Mapping[str, Any],
    feature_names: list[str],
    label_order: list[str],
    feature_metadata: dict[str, Any],
    dataset: str = "maternal_health",
    task_type: str = "multiclass_classification",
    total_counterfactuals: int = 3,
    method: str = "random",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    dice_ml = require("dice_ml", "dice-ml")
    pd = require("pandas")

    if task_type == "regression":
        return {
            "dataset": dataset,
            "patient_id": int(patient[PATIENT_ID]) if PATIENT_ID in patient else None,
            "original_prediction": "continuous_prediction",
            "counterfactual_prediction": None,
            "distance": None,
            "changes": [],
            "validity": "not_generated",
            "feasibility_status": "not_generated",
            "notes": "DiCE counterfactuals are skipped for regression tasks.",
        }

    spec = get_dataset_spec(dataset)
    target_column = spec.target_id
    if not target_column:
        raise ValueError(f"Dataset {dataset} does not have a classification target id.")

    prediction = predict_patient(
        model,
        patient,
        feature_names,
        label_order,
        dataset,
        task_type=task_type,
    )
    desired_class = desired_lower_risk_class(int(prediction["predicted_class_id"]))
    if desired_class == int(prediction["predicted_class_id"]):
        result = {
            "dataset": dataset,
            "patient_id": int(patient[PATIENT_ID]) if PATIENT_ID in patient else None,
            "original_prediction": prediction["predicted_class"],
            "counterfactual_prediction": prediction["predicted_class"],
            "distance": 0.0,
            "changes": [],
            "validity": "not_required",
            "feasibility_status": "not_required",
            "notes": "Patient is already in the lowest risk class.",
        }
        if output_path:
            write_json(result, output_path)
        return result

    data_df = train_df[[*feature_names, target_column]].copy()
    data = dice_ml.Data(
        dataframe=data_df,
        continuous_features=list(feature_names),
        outcome_name=target_column,
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(data, dice_model, method=method)

    query = pd.DataFrame([{feature: patient[feature] for feature in feature_names}])
    counterfactuals = explainer.generate_counterfactuals(
        query,
        total_CFs=total_counterfactuals,
        desired_class=desired_class,
        features_to_vary=mutable_features(feature_metadata),
        permitted_range=permitted_ranges(feature_metadata),
    )
    final_df = counterfactuals.cf_examples_list[0].final_cfs_df
    if final_df is None or final_df.empty:
        raise RuntimeError("DiCE did not return a valid counterfactual.")

    first = final_df.iloc[0].to_dict()
    cf_prediction = predict_patient(
        model,
        first,
        feature_names,
        label_order,
        dataset,
        task_type=task_type,
    )
    validity, problems = validate_counterfactual_changes(patient, first, feature_metadata)

    changes = []
    features = feature_metadata.get("features", feature_metadata)
    for feature in feature_names:
        old = float(patient[feature])
        new = float(first[feature])
        if abs(new - old) <= 1e-8:
            continue
        spec = features.get(feature, {})
        changes.append(
            {
                "feature": feature,
                "from": _json_number(old),
                "to": _json_number(new),
                "mutable": bool(spec.get("mutable", False)),
                "clinical_actionability": spec.get(
                    "clinical_actionability", "unknown"
                ),
            }
        )

    result = {
        "dataset": dataset,
        "patient_id": int(patient[PATIENT_ID]) if PATIENT_ID in patient else None,
        "original_prediction": prediction["predicted_class"],
        "counterfactual_prediction": cf_prediction["predicted_class"],
        "distance": normalized_l1_distance(
            patient, first, permitted_ranges(feature_metadata)
        ),
        "changes": changes,
        "validity": validity,
        "feasibility_status": validity,
        "validation_messages": problems,
    }
    validate_dice_result(result)
    if output_path:
        write_json(result, output_path)
    return result


def _json_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)
