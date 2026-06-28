"""Clinical guideline/PDF chunk loading for vector RAG guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from branch.utils.io import write_json


@dataclass(frozen=True)
class GuidelineChunk:
    chunk_id: str
    source: str
    topic: str
    summary: str
    keywords: list[str]
    feature_directions: dict[str, str]


def default_maternal_guideline_chunks() -> list[GuidelineChunk]:
    source = "curated_maternal_risk_corpus"
    return [
        GuidelineChunk(
            chunk_id="maternal_bp_001",
            source=source,
            topic="Hypertension in pregnancy",
            summary=(
                "Elevated systolic or diastolic blood pressure is a maternal "
                "risk signal and should be reviewed in clinical context."
            ),
            keywords=["blood pressure", "systolic", "diastolic", "hypertension"],
            feature_directions={
                "Systolic BP": "high_value_increases_risk",
                "Diastolic": "high_value_increases_risk",
            },
        ),
        GuidelineChunk(
            chunk_id="maternal_glucose_001",
            source=source,
            topic="Blood glucose in pregnancy",
            summary=(
                "Elevated blood glucose is clinically associated with greater "
                "maternal risk and warrants clinician review."
            ),
            keywords=["blood sugar", "glucose", "BS", "hyperglycemia"],
            feature_directions={"BS": "high_value_increases_risk"},
        ),
        GuidelineChunk(
            chunk_id="maternal_temp_001",
            source=source,
            topic="Body temperature",
            summary=(
                "Elevated body temperature may indicate infection or physiologic "
                "stress and can increase clinical concern."
            ),
            keywords=["body temperature", "body temp", "fever", "infection"],
            feature_directions={"Body Temp": "high_value_increases_risk"},
        ),
        GuidelineChunk(
            chunk_id="maternal_bmi_001",
            source=source,
            topic="Body mass index in pregnancy",
            summary=(
                "Elevated BMI can be associated with higher pregnancy risk and "
                "metabolic complications."
            ),
            keywords=["bmi", "body mass index", "metabolic", "weight"],
            feature_directions={"BMI": "high_value_increases_risk"},
        ),
        GuidelineChunk(
            chunk_id="maternal_history_001",
            source=source,
            topic="Prior pregnancy complications",
            summary=(
                "A history of previous complications can increase clinical "
                "concern in maternal risk assessment."
            ),
            keywords=["previous complications", "history", "maternal complications"],
            feature_directions={"Previous Complications": "presence_increases_risk"},
        ),
        GuidelineChunk(
            chunk_id="maternal_diabetes_001",
            source=source,
            topic="Diabetes and pregnancy risk",
            summary=(
                "Preexisting diabetes and gestational diabetes are clinically "
                "relevant maternal risk factors."
            ),
            keywords=[
                "preexisting diabetes",
                "gestational diabetes",
                "diabetes",
                "glucose",
            ],
            feature_directions={
                "Preexisting Diabetes": "presence_increases_risk",
                "Gestational Diabetes": "presence_increases_risk",
            },
        ),
        GuidelineChunk(
            chunk_id="maternal_mental_health_001",
            source=source,
            topic="Mental health and pregnancy",
            summary=(
                "Mental health concerns can affect maternal well-being and may "
                "support closer clinical review."
            ),
            keywords=["mental health", "well-being", "support"],
            feature_directions={"Mental Health": "presence_increases_risk"},
        ),
        GuidelineChunk(
            chunk_id="maternal_hr_001",
            source=source,
            topic="Maternal heart rate",
            summary=(
                "Markedly elevated heart rate may reflect physiologic stress and "
                "can support higher-risk clinical assessment."
            ),
            keywords=["heart rate", "tachycardia"],
            feature_directions={"Heart Rate": "high_value_increases_risk"},
        ),
        GuidelineChunk(
            chunk_id="maternal_age_001",
            source=source,
            topic="Maternal age",
            summary=(
                "Very young or older maternal age can be associated with increased "
                "pregnancy risk, but age itself is not modifiable."
            ),
            keywords=["age", "maternal age", "older", "young"],
            feature_directions={"Age": "extreme_value_increases_risk"},
        ),
    ]


def save_default_maternal_guidelines(
    output_path: str | Path = "data/external/clinical_guidelines/chunks/maternal_health_guidelines.json",
) -> Path:
    return write_json(
        [asdict(chunk) for chunk in default_maternal_guideline_chunks()], output_path
    )


def load_guideline_chunks_from_directory(
    input_dir: str | Path,
    dataset: str = "all",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[dict]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Clinical PDF corpus folder was not found: {base}. "
            "Create it and add guideline PDFs before building the RAG index."
        )

    paths = sorted(
        [
            *base.rglob("*.pdf"),
            *base.rglob("*.PDF"),
            *base.rglob("*.txt"),
            *base.rglob("*.md"),
        ]
    )
    if not paths:
        raise FileNotFoundError(
            f"No PDF/TXT/MD clinical documents found under {base}. "
            "Add guideline documents before building the RAG index."
        )

    chunks: list[dict] = []
    for path in paths:
        text = _extract_document_text(path)
        if not text.strip():
            continue
        for idx, chunk_text in enumerate(_chunk_text(text, chunk_size, chunk_overlap), start=1):
            chunk_id = f"{_safe_id(path.stem)}_{idx:04d}"
            feature_directions, keywords = infer_feature_directions(
                chunk_text,
                dataset=dataset,
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": str(path),
                    "topic": path.stem.replace("_", " ").replace("-", " "),
                    "summary": chunk_text,
                    "keywords": keywords,
                    "feature_directions": feature_directions,
                    "source_type": "clinical_pdf" if path.suffix.lower() == ".pdf" else "clinical_text",
                }
            )
    if not chunks:
        raise RuntimeError(f"No readable clinical text could be extracted from {base}.")
    return chunks


def save_guideline_chunks(
    chunks: list[dict],
    output_path: str | Path = "data/external/clinical_guidelines/chunks/clinical_guidelines.json",
) -> Path:
    return write_json(chunks, output_path)


def infer_feature_directions(text: str, dataset: str = "all") -> tuple[dict[str, str], list[str]]:
    text_l = text.lower()
    directions: dict[str, str] = {}
    keywords: list[str] = []
    for rule_dataset, feature, rule_keywords, direction in _direction_rules():
        if dataset not in {"all", rule_dataset}:
            continue
        if any(keyword in text_l for keyword in rule_keywords):
            directions[feature] = direction
            keywords.extend([keyword for keyword in rule_keywords if keyword in text_l])
    return directions, sorted(set(keywords))


def _extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF ingestion requires pypdf. Install dependencies with "
                "`py -m pip install -r requirements.txt`."
            ) from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "document"


def _direction_rules() -> list[tuple[str, str, list[str], str]]:
    return [
        ("maternal_health", "Systolic BP", ["systolic", "blood pressure", "hypertension", "preeclampsia"], "high_value_increases_risk"),
        ("maternal_health", "Diastolic", ["diastolic", "blood pressure", "hypertension"], "high_value_increases_risk"),
        ("maternal_health", "BS", ["blood glucose", "blood sugar", "gestational diabetes", "hyperglycemia"], "high_value_increases_risk"),
        ("maternal_health", "Body Temp", ["fever", "body temperature", "infection"], "high_value_increases_risk"),
        ("maternal_health", "BMI", ["bmi", "body mass index", "obesity"], "high_value_increases_risk"),
        ("maternal_health", "Heart Rate", ["heart rate", "tachycardia"], "high_value_increases_risk"),
        ("gallstone", "Total Cholesterol (TC)", ["total cholesterol", "cholesterol", "lipid"], "high_value_increases_risk"),
        ("gallstone", "Low Density Lipoprotein (LDL)", ["ldl", "low density lipoprotein"], "high_value_increases_risk"),
        ("gallstone", "High Density Lipoprotein (HDL)", ["hdl", "high density lipoprotein"], "protective"),
        ("gallstone", "Triglyceride", ["triglyceride", "triglycerides"], "high_value_increases_risk"),
        ("gallstone", "Body Mass Index (BMI)", ["bmi", "body mass index", "obesity"], "high_value_increases_risk"),
        ("gallstone", "Age", ["age", "older adult", "aging"], "extreme_value_increases_risk"),
        ("gallstone", "Glucose", ["glucose", "diabetes", "insulin resistance"], "high_value_increases_risk"),
        ("gallstone", "Alanin Aminotransferaz (ALT)", ["alt", "alanine aminotransferase", "liver enzyme"], "high_value_increases_risk"),
        ("gallstone", "Aspartat Aminotransferaz (AST)", ["ast", "aspartate aminotransferase", "liver enzyme"], "high_value_increases_risk"),
        ("npha", "Phyiscal Health", ["physical health", "self-rated health", "functional status"], "poor_status_increases_risk"),
        ("npha", "Mental Health", ["mental health", "depression", "anxiety"], "poor_status_increases_risk"),
        ("npha", "Trouble Sleeping", ["trouble sleeping", "sleep disturbance", "insomnia"], "high_value_increases_risk"),
        ("npha", "Prescription Sleep Medication", ["sleep medication", "prescription medication", "polypharmacy"], "high_value_increases_risk"),
        ("npha", "Pain Keeps Patient from Sleeping", ["pain", "sleep"], "presence_increases_risk"),
        ("load_diabetes", "bmi", ["bmi", "body mass index", "obesity"], "high_value_increases_risk"),
        ("load_diabetes", "bp", ["blood pressure", "hypertension"], "high_value_increases_risk"),
        ("load_diabetes", "s5", ["triglyceride", "serum", "lipid"], "high_value_increases_risk"),
        ("load_diabetes", "s6", ["glucose", "blood sugar", "glycemic"], "high_value_increases_risk"),
    ]
