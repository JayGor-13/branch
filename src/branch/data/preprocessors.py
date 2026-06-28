"""Preprocessing entry points."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from branch.data.dataset_specs import default_processed_dir, get_dataset_spec, normalize_dataset_name
from branch.data.feature_metadata import write_feature_metadata, write_maternal_feature_metadata
from branch.data.loaders import (
    feature_columns_for_dataframe,
    load_maternal_health_raw,
    load_raw_dataset,
)
from branch.utils.constants import MATERNAL_TARGET, MATERNAL_TARGET_ID
from branch.utils.dependencies import require
from branch.utils.io import ensure_dir, write_json


def preprocess_maternal_health(
    raw_path: str | Path | None = None,
    output_dir: str | Path = "data/processed/maternal_health",
    test_size: float = 0.2,
    seed: int = 42,
):
    train_test_split = require("sklearn.model_selection").train_test_split

    start = perf_counter()
    df = load_maternal_health_raw(raw_path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[MATERNAL_TARGET_ID],
    )

    output = ensure_dir(output_dir)
    train_path = output / "train.csv"
    test_path = output / "test.csv"
    train_df.sort_values("patient_id").to_csv(train_path, index=False)
    test_df.sort_values("patient_id").to_csv(test_path, index=False)

    metadata_path = write_maternal_feature_metadata(output / "feature_metadata.json")
    summary = {
        "dataset": "maternal_health",
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_size": test_size,
        "seed": seed,
        "target_distribution": df[MATERNAL_TARGET_ID].value_counts().sort_index().to_dict(),
        "target_labels": train_df[[MATERNAL_TARGET, MATERNAL_TARGET_ID]]
        .drop_duplicates()
        .sort_values(MATERNAL_TARGET_ID)
        .to_dict(orient="records"),
        "elapsed_sec": round(perf_counter() - start, 4),
        "outputs": {
            "train": str(train_path),
            "test": str(test_path),
            "feature_metadata": str(metadata_path),
        },
    }
    write_json(summary, output / "preprocessing_summary.json")
    return summary


def preprocess_dataset(
    dataset: str,
    raw_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
):
    train_test_split = require("sklearn.model_selection").train_test_split

    normalized = normalize_dataset_name(dataset)
    if normalized == "maternal_health":
        return preprocess_maternal_health(
            raw_path=raw_path,
            output_dir=output_dir or default_processed_dir(normalized),
            test_size=test_size,
            seed=seed,
        )

    start = perf_counter()
    spec = get_dataset_spec(normalized)
    df = load_raw_dataset(normalized, raw_path)
    feature_names = feature_columns_for_dataframe(normalized, df)
    stratify = df[spec.target_id] if spec.is_classification and spec.target_id else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    output = ensure_dir(output_dir or default_processed_dir(normalized))
    train_path = output / "train.csv"
    test_path = output / "test.csv"
    train_df.sort_values("patient_id").to_csv(train_path, index=False)
    test_df.sort_values("patient_id").to_csv(test_path, index=False)

    metadata_path = write_feature_metadata(
        normalized,
        df,
        feature_names,
        output / "feature_metadata.json",
    )
    summary = {
        "dataset": normalized,
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_size": test_size,
        "seed": seed,
        "task_type": spec.task_type,
        "features": feature_names,
        "target": spec.target,
        "target_id": spec.target_id,
        "target_distribution": df[spec.target_id].value_counts().sort_index().to_dict()
        if spec.target_id
        else None,
        "target_labels": [
            {"label": label, "id": idx} for idx, label in enumerate(spec.label_order)
        ],
        "elapsed_sec": round(perf_counter() - start, 4),
        "outputs": {
            "train": str(train_path),
            "test": str(test_path),
            "feature_metadata": str(metadata_path),
        },
    }
    write_json(summary, output / "preprocessing_summary.json")
    return summary
