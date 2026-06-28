from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.data.dataset_specs import default_model_path, default_processed_dir, normalize_dataset_name
from branch.data.feature_metadata import infer_feature_metadata
from branch.data.loaders import feature_columns_for_dataframe, load_processed_split
from branch.explainability.dice_counterfactual import generate_dice_counterfactual
from branch.models.predict import predict_patient
from branch.models.registry import load_model_bundle
from branch.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DiCE counterfactuals.")
    parser.add_argument("--dataset", default="maternal_health")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    processed_dir = Path(args.processed_dir or default_processed_dir(dataset))
    train_df, test_df = load_processed_split(dataset, processed_dir)
    bundle = load_model_bundle(args.model_path or default_model_path(dataset))
    if bundle.task_type == "regression":
        print(f"Skipped DiCE for {dataset}: regression tasks use SHAP-only recourse.")
        return
    metadata_path = processed_dir / "feature_metadata.json"
    metadata = (
        read_json(metadata_path)
        if metadata_path.exists()
        else infer_feature_metadata(
            dataset, train_df, feature_columns_for_dataframe(dataset, train_df)
        )
    )

    saved = 0
    for _, row in test_df.iterrows():
        patient = row.to_dict()
        prediction = predict_patient(
            bundle.model,
            patient,
            bundle.feature_names,
            bundle.label_order,
            dataset,
            task_type=bundle.task_type,
        )
        if not _should_generate_for_prediction(prediction):
            continue
        patient_id = int(patient["patient_id"])
        output_path = (
            Path("artifacts/explanations/dice_json")
            / dataset
            / f"{patient_id}.json"
        )
        generate_dice_counterfactual(
            bundle.model,
            train_df,
            patient,
            bundle.feature_names,
            bundle.label_order,
            metadata,
            dataset,
            task_type=bundle.task_type,
            output_path=output_path,
        )
        saved += 1
        if saved >= args.limit:
            break
    print(f"Saved DiCE counterfactuals for {saved} high-risk patients.")


def _should_generate_for_prediction(prediction: dict) -> bool:
    label = str(prediction.get("predicted_class", "")).lower()
    return any(term in label for term in ["high", "present", "positive"])


if __name__ == "__main__":
    main()
