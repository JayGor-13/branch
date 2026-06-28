"""Shared constants used across BRANCH datasets."""

DATASET_MATERNAL_HEALTH = "maternal_health"
DATASET_GALLSTONE = "gallstone"
DATASET_NPHA = "npha"
DATASET_DIABETES = "load_diabetes"

CLASSIFICATION_DATASETS = {
    DATASET_GALLSTONE,
    DATASET_MATERNAL_HEALTH,
    DATASET_NPHA,
}
REGRESSION_DATASETS = {DATASET_DIABETES}
SUPPORTED_DATASETS = CLASSIFICATION_DATASETS | REGRESSION_DATASETS

MATERNAL_FEATURES = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
    "Heart Rate",
]

MATERNAL_TARGET = "Risk Level"
MATERNAL_TARGET_ID = "RiskLevelId"
PATIENT_ID = "patient_id"

MATERNAL_LABEL_ORDER = ["Low Risk", "Mid Risk", "High Risk"]
MATERNAL_LABEL_TO_ID = {label: idx for idx, label in enumerate(MATERNAL_LABEL_ORDER)}
MATERNAL_ID_TO_LABEL = {idx: label for label, idx in MATERNAL_LABEL_TO_ID.items()}

GALLSTONE_TARGET = "Gallstone Status"
GALLSTONE_TARGET_ID = "GallstoneStatusId"
GALLSTONE_LABEL_ORDER = ["No Gallstone", "Gallstone Present"]

NPHA_ORIGINAL_TARGET = "Number of Doctors Visited"
NPHA_TARGET = "High Utilizer"
NPHA_TARGET_ID = "HighUtilizerId"
NPHA_LABEL_ORDER = ["Not High Utilizer", "High Utilizer"]

DIABETES_TARGET = "DiabetesProgression"

GUARDRAIL_STATUSES = {
    "no_anomaly",
    "possible_anomaly",
    "anomaly_detected",
    "insufficient_guideline_evidence",
    "retrieval_failed",
}
