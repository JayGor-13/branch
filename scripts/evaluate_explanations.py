from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.evaluation.explanation_metrics import (
    clinical_alignment_score,
    completeness,
    explanation_quality_score,
    faithfulness,
)
from branch.agents.llm_client import build_llm_client, build_llm_config
from branch.evaluation.llm_judge import judge_clinical_alignment
from branch.evaluation.ragas_external import RagasRecord, evaluate_ragas_records
from branch.agents.narrative_generator import clean_narrative_for_evaluation, is_valid_narrative
from branch.data.dataset_specs import normalize_dataset_name
from branch.guardrails.alignment_checker import check_clinical_alignment
from branch.guardrails.retriever import retrieve_guidelines
from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.utils.dependencies import require
from branch.utils.env import load_project_env
from branch.utils.io import read_json


MAX_RAGAS_PATIENTS = 10
MAX_SERVER_RPM = 15.0
MIN_REQUEST_INTERVAL_SEC = 60.0 / MAX_SERVER_RPM


def _fallback_alignment_score(guardrail_result: dict) -> float:
    return clinical_alignment_score(guardrail_result)


def main() -> None:
    load_project_env()
    pd = require("pandas")

    parser = argparse.ArgumentParser(description="Evaluate generated explanations.")
    parser.add_argument("--dataset", default="maternal_health")
    parser.add_argument(
        "--trace-dir",
        default=None,
    )
    parser.add_argument(
        "--narrative-dir",
        default=None,
    )
    parser.add_argument(
        "--output-path",
        default="results/metrics/explanation_quality.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of trace files to evaluate.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "Optional method label for Table IV, e.g. BRANCH-Gemma4-26B "
            "or BRANCH-Gemma4-31B. If omitted, the label is "
            "inferred from the trace narrative backend when possible."
        ),
    )
    parser.add_argument(
        "--alignment-mode",
        choices=["guardrail", "llm"],
        default="guardrail",
    )
    parser.add_argument(
        "--quality-mode",
        choices=["local", "ragas"],
        default="local",
        help=(
            "Use `ragas` for the real external RAGAS package, or `local` for "
            "the fast lightweight scorer used in offline smoke tests."
        ),
    )
    parser.add_argument("--ragas-llm-model", default="gemma-4-31b-it")
    parser.add_argument(
        "--ragas-embedding-provider",
        choices=["local", "gemini", "google"],
        default="local",
        help=(
            "Embedding backend used inside RAGAS answer relevancy. Default is "
            "local so Gemini RPM is reserved for evaluator LLM calls."
        ),
    )
    parser.add_argument("--ragas-embedding-model", default="gemini-embedding-001")
    parser.add_argument("--ragas-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--ragas-timeout-sec", type=int, default=300)
    parser.add_argument("--ragas-max-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--ragas-max-retries", type=int, default=1)
    parser.add_argument("--ragas-max-wait", type=int, default=60)
    parser.add_argument("--ragas-answer-strictness", type=int, default=1)
    parser.add_argument(
        "--ragas-llm-min-interval-sec",
        type=float,
        default=5.0,
        help=(
            "Minimum seconds between internal RAGAS evaluator LLM requests. "
            "Use 5 seconds for a conservative 12 RPM pace under a 15 RPM quota."
        ),
    )
    parser.add_argument(
        "--ragas-stop-on-error",
        action="store_true",
        help="Abort if any patient-level RAGAS call fails.",
    )
    parser.add_argument(
        "--ragas-record-delay-sec",
        type=float,
        default=0.0,
        help=(
            "Optional extra pause between patient-level RAGAS evaluations. "
            "RPM is primarily controlled by --ragas-llm-min-interval-sec."
        ),
    )
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--llm-model", default="gemma-4-31b-it")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default=None)
    parser.add_argument("--llm-timeout-sec", type=int, default=120)
    parser.add_argument("--llm-max-tokens", type=int, default=1600)
    parser.add_argument("--no-llm-fallback", action="store_true")
    parser.add_argument("--embedding-provider", default="local")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-api-key-env", default=None)
    parser.add_argument("--embedding-timeout-sec", type=int, default=120)
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    parser.add_argument(
        "--vector-index-path",
        default="artifacts/vector_store/clinical_guidelines",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--retrieval-similarity-threshold", type=float, default=0.0)
    args = parser.parse_args()
    _validate_ragas_config(args)
    dataset = normalize_dataset_name(args.dataset)
    trace_dir = Path(args.trace_dir or f"artifacts/explanations/branch_traces/{dataset}")
    narrative_dir = Path(args.narrative_dir or f"artifacts/explanations/narratives/{dataset}")

    judge_client = None
    if args.alignment_mode == "llm":
        config = build_llm_config(
            provider=args.llm_provider,
            model_name=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
            timeout_sec=args.llm_timeout_sec,
            max_tokens=args.llm_max_tokens,
            fallback_to_template=not args.no_llm_fallback,
        )
        judge_client = build_llm_client(config)
        if judge_client is None:
            raise RuntimeError("LLM alignment mode requires a non-template LLM provider.")
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
    ragas_jobs = []
    trace_paths = sorted(trace_dir.glob("*_trace.json"))
    if args.limit is not None:
        trace_paths = trace_paths[: args.limit]
    for trace_path in trace_paths:
        trace = read_json(trace_path)
        patient_id = trace["patient_id"]
        method_label = args.method or _method_label_from_trace(trace)
        shap_path = Path(trace["shap_result_path"])
        narrative_path = narrative_dir / f"{patient_id}.md"
        shap_result = read_json(shap_path)
        source_narrative = narrative_path.read_text(encoding="utf-8")
        narrative = source_narrative
        guideline_context = retrieve_guidelines(
            dataset,
            trace["prediction"],
            shap_result,
            narrative=narrative,
            top_k=args.retrieval_top_k,
            similarity_threshold=args.retrieval_similarity_threshold,
            vector_index_path=args.vector_index_path,
            embedding_client=embedding_client,
        )
        guardrail_result = check_clinical_alignment(shap_result, guideline_context)
        evaluation_trace = dict(trace)
        evaluation_trace["guideline_context"] = guideline_context
        evaluation_trace["guardrail_result"] = guardrail_result
        evaluation_trace["guardrail_status"] = guardrail_result.get("guardrail_status")
        f_faith = faithfulness(shap_result, narrative)
        f_answer = completeness(shap_result, narrative)
        judge_label = None
        judge_rationale = None
        judge_model = None
        judge_used_llm = False
        try:
            if judge_client is not None:
                judge = judge_clinical_alignment(
                    judge_client, narrative, evaluation_trace, shap_result
                )
                f_align = judge.score
                judge_label = judge.label
                judge_rationale = judge.rationale
                judge_model = judge.model_name
                judge_used_llm = judge.used_llm
            else:
                f_align = _fallback_alignment_score(guardrail_result)
        except Exception:
            if args.no_llm_fallback:
                raise
            f_align = _fallback_alignment_score(guardrail_result)
            judge_label = "fallback_to_guardrail"
            judge_rationale = "Expert LLM judge failed; used RAG fallback score."
        row_index = len(rows)
        rows.append(
            {
                "dataset": dataset,
                "patient_id": patient_id,
                "method": method_label,
                "faithfulness": f_faith,
                "answer_relevancy": f_answer,
                "completeness": f_answer,
                "clinical_alignment": f_align,
                "eqs": explanation_quality_score(f_faith, f_answer, f_align),
                "quality_mode": "local",
                "alignment_mode": args.alignment_mode,
                "judge_label": judge_label,
                "judge_rationale": judge_rationale,
                "judge_model": judge_model,
                "judge_used_llm": judge_used_llm,
                "rag_source": "clinical_guidelines",
                "rag_retrieved_chunks": len(guideline_context.get("retrieved_chunks", [])),
                "rag_retrieved_narratives": 0,
                "rag_backend": (
                    guideline_context.get("retrieval_backend")
                ),
                "embedding_provider": (
                    guideline_context.get("embedding_provider")
                ),
                "embedding_model": (
                    guideline_context.get("embedding_model")
                ),
                "retrieved_patient_id": None,
                "retrieved_narrative_path": None,
                "notes": guardrail_result.get("guardrail_status"),
            }
        )
        if args.quality_mode == "ragas":
            ragas_jobs.append(
                {
                    "row_index": row_index,
                    "record": RagasRecord(
                        user_input=_ragas_user_input(dataset, trace, shap_result),
                        response=_narrative_for_ragas(narrative),
                        retrieved_contexts=_ragas_contexts(guideline_context),
                    ),
                }
            )

    if args.quality_mode == "ragas" and ragas_jobs:
        _print_ragas_preflight(args, dataset, len(ragas_jobs))
        scores = evaluate_ragas_records(
            [job["record"] for job in ragas_jobs],
            llm_model=args.ragas_llm_model,
            embedding_provider=args.ragas_embedding_provider,
            embedding_model=args.ragas_embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            api_key_env=args.ragas_api_key_env,
            timeout_sec=args.ragas_timeout_sec,
            max_workers=args.ragas_max_workers,
            max_retries=args.ragas_max_retries,
            max_wait=args.ragas_max_wait,
            record_delay_sec=args.ragas_record_delay_sec,
            llm_min_interval_sec=args.ragas_llm_min_interval_sec,
            continue_on_error=not args.ragas_stop_on_error,
            answer_relevancy_strictness=args.ragas_answer_strictness,
        )
        for job, score in zip(ragas_jobs, scores):
            row = rows[job["row_index"]]
            row["ragas_error"] = score.error
            row["ragas_evaluator_model"] = args.ragas_llm_model
            row["ragas_embedding_provider"] = args.ragas_embedding_provider
            row["ragas_embedding_model"] = _ragas_embedding_label(
                args.ragas_embedding_provider,
                args.ragas_embedding_model,
                args.embedding_dimensions,
            )
            if score.faithfulness is not None:
                row["faithfulness"] = score.faithfulness
            if score.answer_relevancy is not None:
                row["answer_relevancy"] = score.answer_relevancy
                row["completeness"] = score.answer_relevancy
            if score.faithfulness is None and score.answer_relevancy is None:
                row["quality_mode"] = "ragas_failed"
                row["notes"] = _append_note(row.get("notes"), score.error or "RAGAS metrics unavailable.")
                continue
            if score.faithfulness is not None and score.answer_relevancy is not None:
                row["eqs"] = explanation_quality_score(
                    float(row["faithfulness"]),
                    float(row["answer_relevancy"]),
                    float(row["clinical_alignment"]),
                )
                row["quality_mode"] = "ragas"
            else:
                row["eqs"] = None
                row["quality_mode"] = "ragas_partial"
            if score.error:
                row["notes"] = _append_note(row.get("notes"), score.error)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved explanation quality metrics to {out}")


def _validate_ragas_config(args) -> None:
    if args.quality_mode != "ragas":
        return
    if args.limit is not None and args.limit > MAX_RAGAS_PATIENTS:
        raise SystemExit(
            "RAGAS Table IV evaluation is capped at "
            f"--limit {MAX_RAGAS_PATIENTS}. Received --limit {args.limit}."
        )
    if args.ragas_embedding_provider != "local":
        raise SystemExit(
            "Use --ragas-embedding-provider local for Table IV RAGAS so Gemini "
            "server RPM is reserved for evaluator LLM calls."
        )
    if args.ragas_llm_min_interval_sec < MIN_REQUEST_INTERVAL_SEC:
        raise SystemExit(
            "RAGAS evaluator calls must stay at or below "
            f"{MAX_SERVER_RPM:.0f} RPM. Use --ragas-llm-min-interval-sec "
            f"{MIN_REQUEST_INTERVAL_SEC:.1f} or higher."
        )
    if not os.environ.get(args.ragas_api_key_env) and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit(
            "Missing Gemini API key for RAGAS evaluation. Set "
            f"${args.ragas_api_key_env} before running this command."
        )


def _print_ragas_preflight(args, dataset: str, record_count: int) -> None:
    rpm = 60.0 / args.ragas_llm_min_interval_sec
    print("\n=== RAGAS evaluation preflight ===", flush=True)
    print(f"Dataset: {dataset}", flush=True)
    print(f"Records: {record_count}", flush=True)
    print(f"Evaluator LLM: {args.ragas_llm_model} (~{rpm:.1f} RPM max)", flush=True)
    print(
        f"RAGAS embeddings: {args.ragas_embedding_provider} "
        "(no Gemini embedding requests)",
        flush=True,
    )


def _method_label_from_trace(trace: dict) -> str:
    backend = trace.get("narrative_backend", {})
    model_name = str(backend.get("model_name") or "").lower()
    if "26b" in model_name:
        return "BRANCH-Gemma4-26B"
    if "31b" in model_name:
        return "BRANCH-Gemma4-31B"
    return "BRANCH"


def _ragas_user_input(dataset: str, trace: dict, shap_result: dict) -> str:
    prediction = trace.get("prediction", {})
    features = ", ".join(
        str(item.get("feature")) for item in shap_result.get("features", [])[:5]
    )
    return (
        f"Explain the {dataset} model prediction for patient "
        f"{trace.get('patient_id')}. Include the predicted class or value "
        f"({prediction.get('predicted_class', prediction.get('predicted_value'))}), "
        f"the main SHAP drivers ({features}), any counterfactual pathway when "
        "available, retrieved clinical guideline evidence, guardrail status, "
        "and a clinical caution."
    )


def _ragas_contexts(guideline_context: dict) -> list[str]:
    contexts = []
    for chunk in guideline_context.get("retrieved_chunks", [])[:3]:
        topic = chunk.get("topic", "")
        source = chunk.get("source", "")
        summary = chunk.get("summary", "")
        text = "\n".join(part for part in [topic, source, summary] if part)
        if text:
            contexts.append(_truncate_text(text, max_chars=1200))
    return contexts


def _narrative_for_ragas(narrative: str) -> str:
    cleaned = clean_narrative_for_evaluation(narrative)
    if not is_valid_narrative(cleaned):
        cleaned = _extract_answer_sections(narrative)
    return _truncate_text(cleaned, max_chars=2200)


def _extract_answer_sections(text: str) -> str:
    import re

    cleaned = text.replace("<thought>", "").replace("</thought>", "").strip()
    patterns = [
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?#{1,6}\s*Prediction\b",
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?Prediction\s*:",
    ]
    starts = [
        match.start()
        for pattern in patterns
        for match in re.finditer(pattern, cleaned)
    ]
    if starts:
        cleaned = cleaned[min(starts) :].strip()
    replacements = {
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Prediction\*\*\s*$": "## Prediction",
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Main Model Drivers\*\*\s*$": "## Main Model Drivers",
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Model Evidence Interpretation\*\*\s*$": "## Model Evidence Interpretation",
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Counterfactual Pathway\*\*\s*$": "## Counterfactual Pathway",
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Clinical Evidence Retrieved\*\*\s*$": "## Clinical Evidence Retrieved",
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Guardrail Status\*\*\s*$": "## Guardrail Status",
        r"(?im)^\s*(?:[-*]\s*)?\*\*##\s*Caution\*\*\s*$": "## Caution",
    }
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    if not cleaned.startswith("# BRANCH Explanation"):
        cleaned = "# BRANCH Explanation\n\n" + cleaned
    return cleaned


def _ragas_embedding_label(
    embedding_provider: str, embedding_model: str, embedding_dimensions: int
) -> str:
    if embedding_provider.lower() == "local":
        return f"local-hashing-embedding-{embedding_dimensions}d"
    return embedding_model


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + "\n[truncated]"


def _append_note(existing: object, note: str) -> str:
    existing_text = "" if existing is None else str(existing).strip()
    if not existing_text:
        return note
    return f"{existing_text}; {note}"


if __name__ == "__main__":
    main()
