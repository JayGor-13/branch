"""Model artifact registry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from branch.utils.constants import MATERNAL_FEATURES, MATERNAL_LABEL_ORDER
from branch.utils.dependencies import require
from branch.utils.io import ensure_dir


@dataclass
class ModelBundle:
    model: Any
    feature_names: list[str]
    label_order: list[str]
    task_type: str
    dataset: str


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    joblib = require("joblib")
    out = Path(path)
    ensure_dir(out.parent)
    joblib.dump(
        {
            "model": bundle.model,
            "feature_names": bundle.feature_names,
            "label_order": bundle.label_order,
            "task_type": bundle.task_type,
            "dataset": bundle.dataset,
        },
        out,
    )
    return out


def load_model_bundle(path: str | Path) -> ModelBundle:
    joblib = require("joblib")
    payload = joblib.load(path)
    return ModelBundle(
        model=payload["model"],
        feature_names=list(payload.get("feature_names", MATERNAL_FEATURES)),
        label_order=list(payload.get("label_order", MATERNAL_LABEL_ORDER)),
        task_type=payload.get("task_type", "multiclass_classification"),
        dataset=payload.get("dataset", "maternal_health"),
    )
