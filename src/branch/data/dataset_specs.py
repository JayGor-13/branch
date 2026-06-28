"""Dataset specifications for the BRANCH paper workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from branch.utils.constants import (
    DATASET_DIABETES,
    DATASET_GALLSTONE,
    DATASET_MATERNAL_HEALTH,
    DATASET_NPHA,
    DIABETES_TARGET,
    GALLSTONE_LABEL_ORDER,
    GALLSTONE_TARGET,
    GALLSTONE_TARGET_ID,
    MATERNAL_LABEL_ORDER,
    MATERNAL_TARGET,
    MATERNAL_TARGET_ID,
    NPHA_LABEL_ORDER,
    NPHA_TARGET,
    NPHA_TARGET_ID,
    SUPPORTED_DATASETS,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: str
    target: str
    target_id: str | None
    label_order: list[str]
    raw_dir: Path | None = None
    raw_globs: tuple[str, ...] = ("*.csv", "*.CSV")

    @property
    def is_classification(self) -> bool:
        return self.task_type in {"binary_classification", "multiclass_classification"}

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"


DATASET_SPECS = {
    DATASET_GALLSTONE: DatasetSpec(
        name=DATASET_GALLSTONE,
        task_type="binary_classification",
        target=GALLSTONE_TARGET,
        target_id=GALLSTONE_TARGET_ID,
        label_order=list(GALLSTONE_LABEL_ORDER),
        raw_dir=Path("data/raw/Gallstone_disease"),
    ),
    DATASET_MATERNAL_HEALTH: DatasetSpec(
        name=DATASET_MATERNAL_HEALTH,
        task_type="multiclass_classification",
        target=MATERNAL_TARGET,
        target_id=MATERNAL_TARGET_ID,
        label_order=list(MATERNAL_LABEL_ORDER),
        raw_dir=Path("data/raw/maternal_health"),
    ),
    DATASET_NPHA: DatasetSpec(
        name=DATASET_NPHA,
        task_type="binary_classification",
        target=NPHA_TARGET,
        target_id=NPHA_TARGET_ID,
        label_order=list(NPHA_LABEL_ORDER),
        raw_dir=Path("data/raw/npha"),
    ),
    DATASET_DIABETES: DatasetSpec(
        name=DATASET_DIABETES,
        task_type="regression",
        target=DIABETES_TARGET,
        target_id=None,
        label_order=[],
        raw_dir=None,
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    normalized = normalize_dataset_name(dataset)
    return DATASET_SPECS[normalized]


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower().replace("-", "_")
    aliases = {
        "maternal": DATASET_MATERNAL_HEALTH,
        "mhr": DATASET_MATERNAL_HEALTH,
        "maternal_health_risk": DATASET_MATERNAL_HEALTH,
        "gallstone_disease": DATASET_GALLSTONE,
        "gallstones": DATASET_GALLSTONE,
        "diabetes": DATASET_DIABETES,
        "sklearn_diabetes": DATASET_DIABETES,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}")
    return normalized


def default_processed_dir(dataset: str) -> Path:
    return Path("data/processed") / normalize_dataset_name(dataset)


def default_model_dir(dataset: str) -> Path:
    return Path("artifacts/models") / normalize_dataset_name(dataset)


def default_model_path(dataset: str) -> Path:
    return default_model_dir(dataset) / "xgb_model.pkl"


def default_predictions_path(dataset: str) -> Path:
    return Path("artifacts/predictions") / f"{normalize_dataset_name(dataset)}_predictions.csv"
