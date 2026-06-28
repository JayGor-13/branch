"""Baseline and XGBoost training for BRANCH datasets."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from branch.data.dataset_specs import (
    default_model_dir,
    default_predictions_path,
    get_dataset_spec,
    normalize_dataset_name,
)
from branch.data.loaders import feature_columns_for_dataframe
from branch.evaluation.predictive_metrics import (
    evaluate_multiclass_classifier,
    evaluate_regressor,
)
from branch.models.registry import ModelBundle, save_model_bundle
from branch.utils.constants import (
    MATERNAL_FEATURES,
    MATERNAL_LABEL_ORDER,
    MATERNAL_TARGET,
    MATERNAL_TARGET_ID,
)
from branch.utils.dependencies import require
from branch.utils.io import ensure_dir, write_json


def _build_models(seed: int, num_classes: int):
    LogisticRegression = require("sklearn.linear_model").LogisticRegression
    Pipeline = require("sklearn.pipeline").Pipeline
    RandomForestClassifier = require("sklearn.ensemble").RandomForestClassifier
    StandardScaler = require("sklearn.preprocessing").StandardScaler
    XGBClassifier = require("xgboost").XGBClassifier

    return {
        "LR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        ),
        "XGB-Base": XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def _build_classification_models(seed: int, num_classes: int):
    LogisticRegression = require("sklearn.linear_model").LogisticRegression
    Pipeline = require("sklearn.pipeline").Pipeline
    RandomForestClassifier = require("sklearn.ensemble").RandomForestClassifier
    StandardScaler = require("sklearn.preprocessing").StandardScaler
    XGBClassifier = require("xgboost").XGBClassifier

    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if num_classes == 2 else "mlogloss"
    xgb_kwargs = {
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    if num_classes > 2:
        xgb_kwargs["num_class"] = num_classes

    return {
        "LR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        ),
        "XGB-Base": XGBClassifier(**xgb_kwargs),
    }


def _build_regression_models(seed: int):
    Pipeline = require("sklearn.pipeline").Pipeline
    RandomForestRegressor = require("sklearn.ensemble").RandomForestRegressor
    Ridge = require("sklearn.linear_model").Ridge
    StandardScaler = require("sklearn.preprocessing").StandardScaler
    XGBRegressor = require("xgboost").XGBRegressor

    return {
        "Ridge": Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge())]),
        "RF": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1),
        "XGB-Base": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def train_and_evaluate_maternal(
    train_df,
    test_df,
    output_dir: str | Path = "artifacts/models/maternal_health",
    metrics_path: str | Path = "results/metrics/predictive_metrics.csv",
    predictions_path: str | Path = "artifacts/predictions/maternal_health_predictions.csv",
    seed: int = 42,
):
    pd = require("pandas")

    X_train = train_df[MATERNAL_FEATURES]
    y_train = train_df[MATERNAL_TARGET_ID]
    X_test = test_df[MATERNAL_FEATURES]
    y_test = test_df[MATERNAL_TARGET_ID]
    present_labels = set(train_df[MATERNAL_TARGET]).union(set(test_df[MATERNAL_TARGET]))
    label_order = [label for label in MATERNAL_LABEL_ORDER if label in present_labels]
    task_type = "multiclass_classification" if len(label_order) > 2 else "binary_classification"

    output = ensure_dir(output_dir)
    rows = []
    model_paths = {}
    trained_models = {}

    for method, model in _build_models(seed, num_classes=len(label_order)).items():
        start = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start

        start = perf_counter()
        metrics = evaluate_multiclass_classifier(model, X_test, y_test)
        inference_time = perf_counter() - start
        metrics.update(
            {
                "dataset": "maternal_health",
                "seed": seed,
                "method": method,
                "task_type": task_type,
                "train_time_sec": round(train_time, 4),
                "inference_time_sec": round(inference_time, 4),
            }
        )
        rows.append(metrics)

        filename = {
            "LR": "lr_model.pkl",
            "RF": "rf_model.pkl",
            "XGB-Base": "xgb_model.pkl",
        }[method]
        path = save_model_bundle(
            ModelBundle(
                model=model,
                feature_names=list(MATERNAL_FEATURES),
                label_order=list(label_order),
                task_type=task_type,
                dataset="maternal_health",
            ),
            output / filename,
        )
        model_paths[method] = str(path)
        trained_models[method] = model

    xgb_metrics = next(row for row in rows if row["method"] == "XGB-Base")
    for alias in ["XGB+SHAP-Static", "XGB+DiCE-Static", "BRANCH"]:
        clone = dict(xgb_metrics)
        clone["method"] = alias
        rows.append(clone)

    metrics_df = pd.DataFrame(rows)
    metrics_out = Path(metrics_path)
    ensure_dir(metrics_out.parent)
    if metrics_out.exists():
        previous = pd.read_csv(metrics_out)
        previous = previous[
            ~((previous["dataset"] == "maternal_health") & (previous["seed"] == seed))
        ]
        metrics_df = pd.concat([previous, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_out, index=False)

    xgb_model = trained_models["XGB-Base"]
    probabilities = xgb_model.predict_proba(X_test)
    pred_ids = probabilities.argmax(axis=1)
    pred_df = test_df[["patient_id"]].copy()
    pred_df["true_label_id"] = y_test.to_numpy()
    pred_df["predicted_label_id"] = pred_ids
    for idx, label in enumerate(label_order):
        pred_df[f"prob_{label.replace(' ', '_').lower()}"] = probabilities[:, idx]
    predictions_out = Path(predictions_path)
    ensure_dir(predictions_out.parent)
    pred_df.to_csv(predictions_out, index=False)

    write_json(
        {
            "dataset": "maternal_health",
            "seed": seed,
            "model_paths": model_paths,
            "metrics_path": str(metrics_out),
            "predictions_path": str(predictions_out),
        },
        output / "training_summary.json",
    )
    return metrics_df


def train_and_evaluate_dataset(
    dataset: str,
    train_df,
    test_df,
    output_dir: str | Path | None = None,
    metrics_path: str | Path = "results/metrics/predictive_metrics.csv",
    predictions_path: str | Path | None = None,
    seed: int = 42,
):
    pd = require("pandas")
    np = require("numpy")

    normalized = normalize_dataset_name(dataset)
    if normalized == "maternal_health":
        return train_and_evaluate_maternal(
            train_df,
            test_df,
            output_dir=output_dir or default_model_dir(normalized),
            metrics_path=metrics_path,
            predictions_path=predictions_path or default_predictions_path(normalized),
            seed=seed,
        )

    spec = get_dataset_spec(normalized)
    feature_names = feature_columns_for_dataframe(normalized, train_df)
    X_train = train_df[feature_names]
    X_test = test_df[feature_names]
    y_train = train_df[spec.target_id] if spec.is_classification else train_df[spec.target]
    y_test = test_df[spec.target_id] if spec.is_classification else test_df[spec.target]
    label_order = _present_label_order(spec, train_df, test_df)

    output = ensure_dir(output_dir or default_model_dir(normalized))
    rows = []
    model_paths = {}
    trained_models = {}
    models = (
        _build_classification_models(seed, num_classes=len(label_order))
        if spec.is_classification
        else _build_regression_models(seed)
    )

    for method, model in models.items():
        start = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start

        start = perf_counter()
        metrics = (
            evaluate_multiclass_classifier(model, X_test, y_test)
            if spec.is_classification
            else evaluate_regressor(model, X_test, y_test)
        )
        inference_time = perf_counter() - start
        metrics.update(
            {
                "dataset": normalized,
                "seed": seed,
                "method": method,
                "task_type": spec.task_type,
                "train_time_sec": round(train_time, 4),
                "inference_time_sec": round(inference_time, 4),
            }
        )
        rows.append(metrics)

        filename = {
            "LR": "lr_model.pkl",
            "Ridge": "ridge_model.pkl",
            "RF": "rf_model.pkl",
            "XGB-Base": "xgb_model.pkl",
        }[method]
        path = save_model_bundle(
            ModelBundle(
                model=model,
                feature_names=list(feature_names),
                label_order=list(label_order),
                task_type=spec.task_type,
                dataset=normalized,
            ),
            output / filename,
        )
        model_paths[method] = str(path)
        trained_models[method] = model

    xgb_metrics = next(row for row in rows if row["method"] == "XGB-Base")
    aliases = ["XGB+SHAP-Static", "BRANCH"]
    if spec.is_classification:
        aliases.insert(1, "XGB+DiCE-Static")
    for alias in aliases:
        clone = dict(xgb_metrics)
        clone["method"] = alias
        rows.append(clone)

    metrics_df = pd.DataFrame(rows)
    metrics_out = Path(metrics_path)
    ensure_dir(metrics_out.parent)
    if metrics_out.exists():
        previous = pd.read_csv(metrics_out)
        previous = previous[
            ~((previous["dataset"] == normalized) & (previous["seed"] == seed))
        ]
        metrics_df = pd.concat([previous, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_out, index=False)

    xgb_model = trained_models["XGB-Base"]
    pred_df = test_df[["patient_id"]].copy()
    if spec.is_classification:
        probabilities = xgb_model.predict_proba(X_test)
        pred_ids = probabilities.argmax(axis=1)
        pred_df["true_label_id"] = y_test.to_numpy()
        pred_df["predicted_label_id"] = pred_ids
        for idx, label in enumerate(label_order):
            pred_df[f"prob_{label.replace(' ', '_').lower()}"] = probabilities[:, idx]
    else:
        predictions = np.asarray(xgb_model.predict(X_test))
        pred_df["true_value"] = y_test.to_numpy()
        pred_df["predicted_value"] = predictions
    predictions_out = Path(predictions_path or default_predictions_path(normalized))
    ensure_dir(predictions_out.parent)
    pred_df.to_csv(predictions_out, index=False)

    write_json(
        {
            "dataset": normalized,
            "seed": seed,
            "model_paths": model_paths,
            "metrics_path": str(metrics_out),
            "predictions_path": str(predictions_out),
        },
        output / "training_summary.json",
    )
    return metrics_df[metrics_df["dataset"] == normalized]


def _present_label_order(spec, train_df, test_df) -> list[str]:
    if not spec.is_classification:
        return []
    present_ids = sorted(
        set(train_df[spec.target_id].astype(int)).union(set(test_df[spec.target_id].astype(int)))
    )
    if spec.name == "maternal_health":
        present_labels = set(train_df[spec.target]).union(set(test_df[spec.target]))
        return [label for label in spec.label_order if label in present_labels]
    return [spec.label_order[idx] for idx in present_ids if idx < len(spec.label_order)]
