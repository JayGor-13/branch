"""Predictive metrics used by the paper tables."""

from __future__ import annotations

from typing import Any

from branch.utils.dependencies import require


def evaluate_multiclass_classifier(model: Any, X, y) -> dict[str, float | None]:
    metrics = require("sklearn.metrics")
    np = require("numpy")

    pred = np.asarray(model.predict(X))
    if pred.ndim > 1:
        pred = pred.argmax(axis=1)
    prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    auroc = None
    if prob is not None:
        try:
            unique_classes = np.unique(y)
            if len(unique_classes) == 2 and prob.shape[1] == 2:
                auroc = float(metrics.roc_auc_score(y, prob[:, 1]))
            else:
                auroc = float(
                    metrics.roc_auc_score(y, prob, multi_class="ovr", average="macro")
                )
        except ValueError:
            auroc = None

    return {
        "accuracy": float(metrics.accuracy_score(y, pred)),
        "macro_f1": float(metrics.f1_score(y, pred, average="macro", zero_division=0)),
        "precision": float(metrics.precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(metrics.recall_score(y, pred, average="macro", zero_division=0)),
        "auroc": auroc,
        "mae": None,
        "rmse": None,
        "r2": None,
        "confusion_matrix": np.asarray(metrics.confusion_matrix(y, pred)).tolist(),
    }


def evaluate_regressor(model: Any, X, y) -> dict[str, float | None]:
    metrics = require("sklearn.metrics")
    np = require("numpy")

    pred = np.asarray(model.predict(X))
    return {
        "accuracy": None,
        "macro_f1": None,
        "precision": None,
        "recall": None,
        "auroc": None,
        "mae": float(metrics.mean_absolute_error(y, pred)),
        "rmse": float(metrics.root_mean_squared_error(y, pred)),
        "r2": float(metrics.r2_score(y, pred)),
        "confusion_matrix": None,
    }
