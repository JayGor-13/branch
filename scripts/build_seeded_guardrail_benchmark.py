from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.guardrails.alignment_checker import (
    check_clinical_alignment,
    model_risk_direction,
)
from branch.guardrails.retriever import retrieve_guidelines
from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.utils.dependencies import require


DATASETS = ["gallstone", "maternal_health", "npha", "load_diabetes"]

EXPECTED_DIRECTIONS = {
    "gallstone": {
        "Total Cholesterol (TC)": "increases_risk",
        "Low Density Lipoprotein (LDL)": "increases_risk",
        "High Density Lipoprotein (HDL)": "decreases_risk",
        "Triglyceride": "increases_risk",
        "Body Mass Index (BMI)": "increases_risk",
        "Age": "increases_risk",
        "Glucose": "increases_risk",
        "Alanin Aminotransferaz (ALT)": "increases_risk",
        "Aspartat Aminotransferaz (AST)": "increases_risk",
    },
    "maternal_health": {
        "Systolic BP": "increases_risk",
        "Diastolic": "increases_risk",
        "BS": "increases_risk",
        "Body Temp": "increases_risk",
        "BMI": "increases_risk",
        "Heart Rate": "increases_risk",
        "Previous Complications": "increases_risk",
        "Preexisting Diabetes": "increases_risk",
        "Gestational Diabetes": "increases_risk",
        "Mental Health": "increases_risk",
    },
    "npha": {
        "Phyiscal Health": "increases_risk",
        "Mental Health": "increases_risk",
        "Trouble Sleeping": "increases_risk",
        "Prescription Sleep Medication": "increases_risk",
        "Pain Keeps Patient from Sleeping": "increases_risk",
        "Medication Keeps Patient from Sleeping": "increases_risk",
        "Bathroom Needs Keeps Patient from Sleeping": "increases_risk",
    },
    "load_diabetes": {
        "bmi": "increases_risk",
        "bp": "increases_risk",
        "s5": "increases_risk",
        "s6": "increases_risk",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a seeded anomaly benchmark for guardrail evaluation."
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--shap-root", default="artifacts/explanations/shap_json")
    parser.add_argument(
        "--vector-index-path", default="artifacts/vector_store/clinical_guidelines"
    )
    parser.add_argument(
        "--output-path", default="results/metrics/guardrail_seeded_benchmark.csv"
    )
    parser.add_argument("--cases-per-class", type=int, default=30)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--retrieval-similarity-threshold", type=float, default=0.0)
    parser.add_argument("--embedding-provider", default="local")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-api-key-env", default=None)
    parser.add_argument("--embedding-timeout-sec", type=int, default=120)
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    args = parser.parse_args()

    datasets = DATASETS if args.datasets == "all" else [
        item.strip() for item in args.datasets.split(",") if item.strip()
    ]
    embedding_config = build_embedding_config(
        provider=args.embedding_provider,
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key_env=args.embedding_api_key_env,
        timeout_sec=args.embedding_timeout_sec,
        dimensions=args.embedding_dimensions,
    )
    embedding_client = build_embedding_client(embedding_config)

    rows = []
    for dataset in datasets:
        rows.extend(
            build_dataset_cases(
                dataset=dataset,
                shap_dir=Path(args.shap_root) / dataset,
                vector_index_path=args.vector_index_path,
                embedding_client=embedding_client,
                cases_per_class=args.cases_per_class,
                retrieval_top_k=args.retrieval_top_k,
                retrieval_similarity_threshold=args.retrieval_similarity_threshold,
            )
        )

    pd = require("pandas")
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved seeded guardrail benchmark to {out}")


def build_dataset_cases(
    dataset: str,
    shap_dir: Path,
    vector_index_path: str | Path,
    embedding_client: Any,
    cases_per_class: int,
    retrieval_top_k: int,
    retrieval_similarity_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    selected = []
    for path in sorted(shap_dir.glob("*.json")):
        shap_result = json.loads(path.read_text(encoding="utf-8"))
        item = select_salient_feature(dataset, shap_result)
        if item is None:
            continue
        selected.append((path, shap_result, item))
        if len(selected) >= cases_per_class:
            break

    for path, shap_result, item in selected:
        rows.append(
            run_seeded_case(
                shap_path=path,
                shap_result=shap_result,
                feature_item=item,
                true_anomaly=False,
                vector_index_path=vector_index_path,
                embedding_client=embedding_client,
                retrieval_top_k=retrieval_top_k,
                retrieval_similarity_threshold=retrieval_similarity_threshold,
            )
        )
        rows.append(
            run_seeded_case(
                shap_path=path,
                shap_result=shap_result,
                feature_item=item,
                true_anomaly=True,
                vector_index_path=vector_index_path,
                embedding_client=embedding_client,
                retrieval_top_k=retrieval_top_k,
                retrieval_similarity_threshold=retrieval_similarity_threshold,
            )
        )
    return rows


def run_seeded_case(
    shap_path: Path,
    shap_result: dict[str, Any],
    feature_item: dict[str, Any],
    true_anomaly: bool,
    vector_index_path: str | Path,
    embedding_client: Any,
    retrieval_top_k: int,
    retrieval_similarity_threshold: float,
) -> dict[str, Any]:
    seeded = build_seeded_shap_result(shap_result, feature_item, true_anomaly)
    guideline_context = retrieve_guidelines(
        seeded["dataset"],
        seeded["prediction"],
        seeded,
        top_k=retrieval_top_k,
        similarity_threshold=retrieval_similarity_threshold,
        vector_index_path=vector_index_path,
        embedding_client=embedding_client,
    )
    guardrail = check_clinical_alignment(seeded, guideline_context)
    check = guardrail.get("alignment_checks", [{}])[0]
    predicted_anomaly = guardrail.get("guardrail_status") == "anomaly_detected"
    return {
        "dataset": seeded["dataset"],
        "patient_id": seeded["patient_id"],
        "case_id": seeded["case_id"],
        "case_type": "seeded_anomaly" if true_anomaly else "control",
        "feature": feature_item["feature"],
        "feature_value": feature_item.get("value"),
        "original_shap": feature_item.get("shap"),
        "seeded_shap": seeded["features"][0]["shap"],
        "expected_clinical_direction": seeded["expected_clinical_direction"],
        "model_direction": check.get("model_direction"),
        "guideline_direction": check.get("guideline_direction"),
        "alignment": check.get("alignment"),
        "true_anomaly": int(true_anomaly),
        "predicted_anomaly": int(predicted_anomaly),
        "guardrail_status": guardrail.get("guardrail_status"),
        "retrieved_chunks": len(guideline_context.get("retrieved_chunks", [])),
        "evidence_chunk_ids": ";".join(check.get("evidence_chunk_ids", [])),
        "source_shap_path": str(shap_path),
    }


def build_seeded_shap_result(
    shap_result: dict[str, Any],
    feature_item: dict[str, Any],
    true_anomaly: bool,
) -> dict[str, Any]:
    dataset = shap_result["dataset"]
    feature = feature_item["feature"]
    expected_direction = EXPECTED_DIRECTIONS[dataset][feature]
    predicted_class = shap_result.get("predicted_class", "")
    magnitude = max(abs(float(feature_item.get("shap", 0.0))), 0.1)
    sign = sign_for_model_direction(predicted_class, expected_direction)
    if true_anomaly:
        sign *= -1
    seeded_item = copy.deepcopy(feature_item)
    seeded_item["rank"] = 1
    seeded_item["shap"] = sign * magnitude
    seeded_item["direction"] = (
        "increases_prediction" if seeded_item["shap"] >= 0 else "decreases_prediction"
    )

    seeded = copy.deepcopy(shap_result)
    seeded["features"] = [seeded_item]
    seeded["top_k"] = 1
    seeded["expected_clinical_direction"] = expected_direction
    seeded["case_id"] = (
        f"{dataset}_{seeded.get('patient_id')}_{feature.replace(' ', '_')}_"
        f"{'anomaly' if true_anomaly else 'control'}"
    )
    return seeded


def select_salient_feature(
    dataset: str, shap_result: dict[str, Any]
) -> dict[str, Any] | None:
    expected = EXPECTED_DIRECTIONS.get(dataset, {})
    for item in shap_result.get("features", []):
        if item.get("feature") in expected:
            return item
    return None


def sign_for_model_direction(predicted_class: str, desired_direction: str) -> int:
    positive_direction = model_risk_direction(1.0, predicted_class)
    return 1 if positive_direction == desired_direction else -1


if __name__ == "__main__":
    main()
