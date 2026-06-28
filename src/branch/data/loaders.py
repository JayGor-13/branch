"""Dataset loaders for BRANCH tabular datasets."""

from __future__ import annotations

from pathlib import Path

from branch.data.dataset_specs import (
    default_processed_dir,
    get_dataset_spec,
    normalize_dataset_name,
)
from branch.utils.constants import (
    DATASET_DIABETES,
    DATASET_GALLSTONE,
    MATERNAL_FEATURES,
    MATERNAL_LABEL_ORDER,
    MATERNAL_TARGET,
    MATERNAL_TARGET_ID,
    NPHA_ORIGINAL_TARGET,
    NPHA_TARGET,
    NPHA_TARGET_ID,
    PATIENT_ID,
)
from branch.utils.dependencies import require


_COLUMN_ALIASES = {
    "age": "Age",
    "systolicbp": "Systolic BP",
    "systolic bp": "Systolic BP",
    "systolic_bp": "Systolic BP",
    "systolic blood pressure": "Systolic BP",
    "diastolic": "Diastolic",
    "diastolicbp": "Diastolic",
    "diastolic bp": "Diastolic",
    "diastolic_bp": "Diastolic",
    "diastolic blood pressure": "Diastolic",
    "bs": "BS",
    "bloodsugar": "BS",
    "blood_sugar": "BS",
    "blood glucose": "BS",
    "glucose": "BS",
    "bodytemp": "Body Temp",
    "body temp": "Body Temp",
    "body_temp": "Body Temp",
    "body temperature": "Body Temp",
    "bmi": "BMI",
    "previouscomplications": "Previous Complications",
    "previous complications": "Previous Complications",
    "previous_complications": "Previous Complications",
    "preexistingdiabetes": "Preexisting Diabetes",
    "preexisting diabetes": "Preexisting Diabetes",
    "preexisting_diabetes": "Preexisting Diabetes",
    "pre existing diabetes": "Preexisting Diabetes",
    "gestationaldiabetes": "Gestational Diabetes",
    "gestational diabetes": "Gestational Diabetes",
    "gestational_diabetes": "Gestational Diabetes",
    "mentalhealth": "Mental Health",
    "mental health": "Mental Health",
    "mental_health": "Mental Health",
    "heartrate": "Heart Rate",
    "heart rate": "Heart Rate",
    "heart_rate": "Heart Rate",
    "risklevel": "Risk Level",
    "risk level": "Risk Level",
    "risk_level": "Risk Level",
}

_TARGET_ALIASES = {
    "low risk": "Low Risk",
    "low": "Low Risk",
    "mid risk": "Mid Risk",
    "medium risk": "Mid Risk",
    "mid": "Mid Risk",
    "high risk": "High Risk",
    "high": "High Risk",
}


def find_maternal_raw_path(raw_path: str | Path | None = None) -> Path:
    return find_raw_csv("maternal_health", raw_path)


def find_raw_csv(dataset: str, raw_path: str | Path | None = None) -> Path:
    spec = get_dataset_spec(dataset)
    if spec.name == DATASET_DIABETES:
        raise ValueError("load_diabetes is loaded from scikit-learn, not a raw CSV.")

    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Raw dataset not found: {candidate}")

    if spec.raw_dir is None:
        raise ValueError(f"Dataset {dataset} does not have a raw CSV directory.")

    for pattern in spec.raw_globs:
        matches = sorted(spec.raw_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Raw CSV for {spec.name} was not found. Place a CSV under "
        f"{spec.raw_dir} or pass --raw-path."
    )


def _canonical_column_name(column: str) -> str:
    key = str(column).strip().lower().replace("-", " ").replace("_", " ")
    compact = key.replace(" ", "")
    return _COLUMN_ALIASES.get(compact, _COLUMN_ALIASES.get(key, str(column).strip()))


def normalize_maternal_dataframe(df):
    pd = require("pandas")

    normalized = df.rename(columns={column: _canonical_column_name(column) for column in df.columns})
    required = [*MATERNAL_FEATURES, MATERNAL_TARGET]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Maternal Health dataset is missing required columns: {missing}")

    out = normalized.copy()
    for feature in MATERNAL_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")

    raw_target = out[MATERNAL_TARGET]
    missing_target = raw_target.isna() | raw_target.astype(str).str.strip().eq("")
    target_text = raw_target.astype(str).str.strip().str.lower()
    mapped_target = target_text.map(_TARGET_ALIASES)
    unknown_target = mapped_target.isna() & ~missing_target
    if unknown_target.any():
        bad_values = sorted(set(target_text[unknown_target].tolist()))
        raise ValueError(f"Unrecognized maternal risk labels: {bad_values}")

    out = out.loc[~missing_target].copy()
    out[MATERNAL_TARGET] = mapped_target.loc[~missing_target]
    for feature in MATERNAL_FEATURES:
        if out[feature].isna().any():
            median_value = out[feature].median()
            if pd.isna(median_value):
                raise ValueError(f"Feature has no usable numeric values: {feature}")
            out[feature] = out[feature].fillna(median_value)

    if out[MATERNAL_FEATURES].isna().any().any():
        missing_counts = out[MATERNAL_FEATURES].isna().sum()
        raise ValueError(f"Missing numeric feature values detected: {missing_counts.to_dict()}")

    label_order = maternal_label_order_for_labels(out[MATERNAL_TARGET])
    label_to_id = {label: idx for idx, label in enumerate(label_order)}
    out[MATERNAL_TARGET_ID] = out[MATERNAL_TARGET].map(label_to_id).astype(int)
    if PATIENT_ID not in out.columns:
        out.insert(0, PATIENT_ID, range(len(out)))

    columns = [PATIENT_ID, *MATERNAL_FEATURES, MATERNAL_TARGET, MATERNAL_TARGET_ID]
    return out[columns].reset_index(drop=True)


def maternal_label_order_for_labels(labels) -> list[str]:
    present = set(labels)
    label_order = [label for label in MATERNAL_LABEL_ORDER if label in present]
    if not label_order:
        raise ValueError("No recognized maternal risk labels were found.")
    return label_order


def load_maternal_health_raw(raw_path: str | Path | None = None):
    pd = require("pandas")
    path = find_maternal_raw_path(raw_path)
    return normalize_maternal_dataframe(pd.read_csv(path))


def normalize_gallstone_dataframe(df):
    pd = require("pandas")

    normalized = df.rename(columns={column: str(column).strip() for column in df.columns})
    target = "Gallstone Status"
    if target not in normalized.columns:
        raise ValueError(f"Gallstone dataset is missing target column: {target}")

    out = normalized.copy()
    out = out.loc[~out[target].isna()].copy()
    feature_names = [column for column in out.columns if column != target]
    for column in feature_names:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any():
            median_value = out[column].median()
            if pd.isna(median_value):
                raise ValueError(f"Feature has no usable numeric values: {column}")
            out[column] = out[column].fillna(median_value)

    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.loc[~out[target].isna()].copy()
    out["GallstoneStatusId"] = out[target].astype(int)
    out[target] = out["GallstoneStatusId"].map(
        {0: "No Gallstone", 1: "Gallstone Present"}
    )
    if out[target].isna().any():
        bad_values = sorted(set(out["GallstoneStatusId"].tolist()))
        raise ValueError(f"Unrecognized gallstone labels: {bad_values}")
    if PATIENT_ID not in out.columns:
        out.insert(0, PATIENT_ID, range(len(out)))
    return out[[PATIENT_ID, *feature_names, target, "GallstoneStatusId"]].reset_index(
        drop=True
    )


def load_gallstone_raw(raw_path: str | Path | None = None):
    pd = require("pandas")
    path = find_raw_csv("gallstone", raw_path)
    return normalize_gallstone_dataframe(pd.read_csv(path))


def normalize_npha_dataframe(df):
    pd = require("pandas")

    normalized = df.rename(columns={column: str(column).strip() for column in df.columns})
    if NPHA_ORIGINAL_TARGET not in normalized.columns:
        raise ValueError(f"NPHA dataset is missing target column: {NPHA_ORIGINAL_TARGET}")

    out = normalized.copy()
    out = out.loc[~out[NPHA_ORIGINAL_TARGET].isna()].copy()
    feature_names = [column for column in out.columns if column != NPHA_ORIGINAL_TARGET]
    for column in [*feature_names, NPHA_ORIGINAL_TARGET]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any():
            median_value = out[column].median()
            if pd.isna(median_value):
                raise ValueError(f"Column has no usable numeric values: {column}")
            out[column] = out[column].fillna(median_value)

    # The local NPHA CSV encodes visits as an ordinal category (1/2/3), not raw counts.
    # We define high utilization as the highest observed visit category.
    high_category = out[NPHA_ORIGINAL_TARGET].max()
    out[NPHA_TARGET_ID] = (out[NPHA_ORIGINAL_TARGET] >= high_category).astype(int)
    out[NPHA_TARGET] = out[NPHA_TARGET_ID].map(
        {0: "Not High Utilizer", 1: "High Utilizer"}
    )
    if PATIENT_ID not in out.columns:
        out.insert(0, PATIENT_ID, range(len(out)))
    return out[
        [PATIENT_ID, *feature_names, NPHA_ORIGINAL_TARGET, NPHA_TARGET, NPHA_TARGET_ID]
    ].reset_index(drop=True)


def load_npha_raw(raw_path: str | Path | None = None):
    pd = require("pandas")
    path = find_raw_csv("npha", raw_path)
    return normalize_npha_dataframe(pd.read_csv(path))


def load_diabetes_raw():
    pd = require("pandas")
    load_diabetes = require("sklearn.datasets").load_diabetes

    dataset = load_diabetes()
    out = pd.DataFrame(dataset.data, columns=list(dataset.feature_names))
    out.insert(0, PATIENT_ID, range(len(out)))
    out["DiabetesProgression"] = dataset.target
    return out


def load_raw_dataset(dataset: str, raw_path: str | Path | None = None):
    normalized = normalize_dataset_name(dataset)
    if normalized == "maternal_health":
        return load_maternal_health_raw(raw_path)
    if normalized == DATASET_GALLSTONE:
        return load_gallstone_raw(raw_path)
    if normalized == "npha":
        return load_npha_raw(raw_path)
    if normalized == DATASET_DIABETES:
        return load_diabetes_raw()
    raise ValueError(f"Unsupported dataset: {dataset}")


def feature_columns_for_dataframe(dataset: str, df) -> list[str]:
    spec = get_dataset_spec(dataset)
    excluded = {PATIENT_ID, spec.target}
    if spec.target_id:
        excluded.add(spec.target_id)
    if spec.name == "npha":
        excluded.add(NPHA_ORIGINAL_TARGET)
    return [column for column in df.columns if column not in excluded]


def load_processed_maternal_split(processed_dir: str | Path = "data/processed/maternal_health"):
    return load_processed_split("maternal_health", processed_dir)


def load_processed_split(
    dataset: str,
    processed_dir: str | Path | None = None,
):
    pd = require("pandas")
    spec = get_dataset_spec(dataset)
    base = Path(processed_dir) if processed_dir else default_processed_dir(spec.name)
    train_path = base / "train.csv"
    test_path = base / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed {spec.name} split is missing. Run "
            f"`py .\\scripts\\preprocess_all.py --dataset {spec.name}` first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)
