from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.data.dataset_specs import default_model_path, default_processed_dir, normalize_dataset_name
from branch.data.loaders import load_processed_split
from branch.explainability.shap_explainer import explain_patient_shap
from branch.models.registry import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations.")
    parser.add_argument("--dataset", default="maternal_health")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    _, test_df = load_processed_split(dataset, args.processed_dir or default_processed_dir(dataset))
    bundle = load_model_bundle(args.model_path or default_model_path(dataset))
    for _, row in test_df.head(args.limit).iterrows():
        patient = row.to_dict()
        patient_id = int(patient["patient_id"])
        output_path = (
            Path("artifacts/explanations/shap_json")
            / dataset
            / f"{patient_id}.json"
        )
        explain_patient_shap(
            bundle.model,
            patient,
            bundle.feature_names,
            bundle.label_order,
            dataset,
            task_type=bundle.task_type,
            top_k=args.top_k,
            output_path=output_path,
        )
    print(f"Saved SHAP explanations for {min(args.limit, len(test_df))} patients.")


if __name__ == "__main__":
    main()
