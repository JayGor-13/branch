from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.utils.env import load_project_env


DEFAULT_DATASETS = ["gallstone", "maternal_health", "npha"]
MAX_TABLE_IV_PATIENTS = 10
MAX_SERVER_RPM = 15.0
MIN_REQUEST_INTERVAL_SEC = 60.0 / MAX_SERVER_RPM
VARIANTS = [
    (
        "BRANCH-Gemma4-26B",
        "BRANCH_GEMMA4_26B_MODEL",
        "gemma-4-26b-a4b-it",
    ),
    (
        "BRANCH-Gemma4-31B",
        "BRANCH_GEMMA4_31B_MODEL",
        "gemma-4-31b-it",
    ),
]


def main() -> None:
    load_project_env()

    parser = argparse.ArgumentParser(
        description=(
            "Run BRANCH Gemini API narrative variants and save Table IV metrics. "
            "The paper-facing method labels stay BRANCH-Gemma4-* while the actual "
            "Gemini API model ids are configurable."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Classification datasets to evaluate. Use: gallstone maternal_health npha",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=int, default=120)
    parser.add_argument("--llm-max-tokens", type=int, default=1600)
    parser.add_argument("--llm-max-retries", type=int, default=5)
    parser.add_argument("--llm-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument(
        "--llm-request-delay-sec",
        type=float,
        default=5.0,
        help="Pause after each generated BRANCH narrative to stay below RPM quotas.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=["ragas", "local"],
        default="ragas",
        help="Use real external RAGAS by default; use local only for smoke tests.",
    )
    parser.add_argument("--ragas-llm-model", default="gemma-4-31b-it")
    parser.add_argument(
        "--ragas-embedding-provider",
        choices=["local", "gemini", "google"],
        default="local",
        help="Use local RAGAS embeddings by default to avoid extra Gemini RPM.",
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
        help="Minimum seconds between internal RAGAS evaluator LLM calls.",
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
    parser.add_argument("--allow-template-fallback", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["26b", "31b", "all"],
        default=["26b", "31b"],
        help="Which Gemini variants to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patients whose trace and narrative already exist.",
    )
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
    parser.add_argument(
        "--variant-artifact-root",
        default="artifacts/gemini_variants",
        help=(
            "Root folder for variant-specific BRANCH traces and narratives. "
            "This prevents 26B and 31B runs from overwriting each other."
        ),
    )
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--retrieval-similarity-threshold", type=float, default=0.0)
    parser.add_argument(
        "--skip-table-generation",
        action="store_true",
        help="Only write explanation_quality_*.csv files; do not regenerate LaTeX tables.",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Do not call Gemini ListModels before running the experiment.",
    )
    parser.add_argument(
        "--list-gemini-models",
        action="store_true",
        help="Print Gemini text-generation model ids available to this API key and exit.",
    )
    parser.add_argument(
        "--model-26b",
        default=None,
        help="Actual Gemini API model id for BRANCH-Gemma4-26B.",
    )
    parser.add_argument(
        "--model-31b",
        default=None,
        help="Actual Gemini API model id for BRANCH-Gemma4-31B.",
    )
    args = parser.parse_args()

    variant_models = _resolve_variant_models(args)
    _validate_table_iv_config(args, variant_models)
    _validate_api_key(args.llm_api_key_env)
    if args.list_gemini_models:
        for model_id in _list_gemini_generation_models(args.llm_api_key_env):
            print(model_id)
        return
    if not args.skip_model_validation:
        _validate_gemini_models(args.llm_api_key_env, variant_models)
    _print_preflight(args, variant_models)

    for method_label, model_id in variant_models:
        artifact_root = _variant_artifact_root(args.variant_artifact_root, method_label)
        print(f"\n=== {method_label} -> Gemini model: {model_id} ===", flush=True)
        for dataset in args.datasets:
            print(f"[{method_label}] Running BRANCH explanations for {dataset}", flush=True)
            _run(
                [
                    sys.executable,
                    "scripts/run_branch_explanations.py",
                    "--dataset",
                    dataset,
                    "--limit",
                    str(args.limit),
                    "--artifact-root",
                    str(artifact_root),
                    "--llm-provider",
                    args.llm_provider,
                    "--llm-model",
                    model_id,
                    "--llm-api-key-env",
                    args.llm_api_key_env,
                    "--llm-timeout-sec",
                    str(args.llm_timeout_sec),
                    "--llm-max-tokens",
                    str(args.llm_max_tokens),
                    "--llm-max-retries",
                    str(args.llm_max_retries),
                    "--llm-retry-backoff-sec",
                    str(args.llm_retry_backoff_sec),
                    "--llm-request-delay-sec",
                    str(args.llm_request_delay_sec),
                    "--embedding-provider",
                    args.embedding_provider,
                    "--embedding-timeout-sec",
                    str(args.embedding_timeout_sec),
                    "--embedding-dimensions",
                    str(args.embedding_dimensions),
                    "--vector-index-path",
                    args.vector_index_path,
                    "--retrieval-top-k",
                    str(args.retrieval_top_k),
                    "--retrieval-similarity-threshold",
                    str(args.retrieval_similarity_threshold),
                    *(_optional_arg("--llm-base-url", args.llm_base_url)),
                    *(_optional_arg("--embedding-model", args.embedding_model)),
                    *(_optional_arg("--embedding-base-url", args.embedding_base_url)),
                    *(
                        _optional_arg(
                            "--embedding-api-key-env", args.embedding_api_key_env
                        )
                    ),
                    *(["--no-llm-fallback"] if not args.allow_template_fallback else []),
                    *(["--skip-existing"] if args.resume else []),
                ]
            )

            output_path = _metrics_output_path(method_label, dataset)
            print(
                f"[{method_label}] Evaluating explanation quality for {dataset}",
                flush=True,
            )
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_explanations.py",
                    "--dataset",
                    dataset,
                    "--method",
                    method_label,
                    "--quality-mode",
                    args.quality_mode,
                    "--limit",
                    str(args.limit),
                    "--ragas-llm-model",
                    args.ragas_llm_model,
                    "--ragas-embedding-provider",
                    args.ragas_embedding_provider,
                    *(
                        _optional_arg(
                            "--ragas-embedding-model",
                            args.ragas_embedding_model,
                        )
                        if args.ragas_embedding_provider != "local"
                        else []
                    ),
                    "--ragas-api-key-env",
                    args.ragas_api_key_env,
                    "--ragas-timeout-sec",
                    str(args.ragas_timeout_sec),
                    "--ragas-max-retries",
                    str(args.ragas_max_retries),
                    "--ragas-max-wait",
                    str(args.ragas_max_wait),
                    "--ragas-answer-strictness",
                    str(args.ragas_answer_strictness),
                    "--ragas-llm-min-interval-sec",
                    str(args.ragas_llm_min_interval_sec),
                    "--ragas-record-delay-sec",
                    str(args.ragas_record_delay_sec),
                    *(["--ragas-stop-on-error"] if args.ragas_stop_on_error else []),
                    "--trace-dir",
                    str(artifact_root / "explanations" / "branch_traces" / dataset),
                    "--narrative-dir",
                    str(artifact_root / "explanations" / "narratives" / dataset),
                    "--output-path",
                    str(output_path),
                    "--embedding-provider",
                    args.embedding_provider,
                    "--embedding-timeout-sec",
                    str(args.embedding_timeout_sec),
                    "--embedding-dimensions",
                    str(args.embedding_dimensions),
                    "--vector-index-path",
                    args.vector_index_path,
                    "--retrieval-top-k",
                    str(args.retrieval_top_k),
                    "--retrieval-similarity-threshold",
                    str(args.retrieval_similarity_threshold),
                    *(_optional_arg("--embedding-model", args.embedding_model)),
                    *(_optional_arg("--embedding-base-url", args.embedding_base_url)),
                    *(
                        _optional_arg(
                            "--embedding-api-key-env", args.embedding_api_key_env
                        )
                    ),
                ]
            )

    if not args.skip_table_generation:
        print("\n=== Regenerating paper tables ===", flush=True)
        _run(
            [
                sys.executable,
                "scripts/generate_paper_tables.py",
                "--table-iv-sample-size",
                str(args.limit),
                "--variant-artifact-root",
                args.variant_artifact_root,
            ]
        )


def _resolve_variant_models(args) -> list[tuple[str, str]]:
    cli_values = {
        "BRANCH-Gemma4-26B": args.model_26b,
        "BRANCH-Gemma4-31B": args.model_31b,
    }
    selected = (
        {"26b", "31b"}
        if "all" in args.variants
        else {variant.lower() for variant in args.variants}
    )
    resolved = []
    for method_label, env_name, default_model in VARIANTS:
        if _variant_key(method_label) not in selected:
            continue
        model_id = cli_values[method_label] or os.environ.get(env_name) or default_model
        resolved.append((method_label, model_id))
    return resolved


def _validate_table_iv_config(args, variant_models: list[tuple[str, str]]) -> None:
    if args.quality_mode != "ragas":
        return
    if args.limit > MAX_TABLE_IV_PATIENTS:
        raise SystemExit(
            "Table IV RAGAS runs are capped at "
            f"--limit {MAX_TABLE_IV_PATIENTS} for this experiment. "
            f"Received --limit {args.limit}."
        )
    if args.ragas_embedding_provider != "local":
        raise SystemExit(
            "Table IV RAGAS must use --ragas-embedding-provider local so Gemini "
            "server RPM is reserved for evaluator LLM calls."
        )
    if args.ragas_llm_min_interval_sec < MIN_REQUEST_INTERVAL_SEC:
        raise SystemExit(
            "Table IV RAGAS must keep evaluator LLM calls at or below "
            f"{MAX_SERVER_RPM:.0f} RPM. Use --ragas-llm-min-interval-sec "
            f"{MIN_REQUEST_INTERVAL_SEC:.1f} or higher."
        )
    if args.llm_provider.lower() in {"gemini", "google", "gemma"}:
        if args.llm_request_delay_sec < MIN_REQUEST_INTERVAL_SEC:
            raise SystemExit(
                "Gemini narrative generation must stay at or below "
                f"{MAX_SERVER_RPM:.0f} RPM. Use --llm-request-delay-sec "
                f"{MIN_REQUEST_INTERVAL_SEC:.1f} or higher."
            )
    if not variant_models:
        raise SystemExit("No Gemini variants selected. Use --variants 26b 31b.")


def _print_preflight(args, variant_models: list[tuple[str, str]]) -> None:
    if args.quality_mode != "ragas":
        return
    evaluator_rpm = 60.0 / args.ragas_llm_min_interval_sec
    generation_rpm = (
        60.0 / args.llm_request_delay_sec if args.llm_request_delay_sec > 0 else None
    )
    print("\n=== Table IV RAGAS preflight ===", flush=True)
    print(f"Datasets: {', '.join(args.datasets)}", flush=True)
    print(f"Patients per dataset: {args.limit}", flush=True)
    print(
        "Variants: "
        + ", ".join(f"{method} -> {model}" for method, model in variant_models),
        flush=True,
    )
    print(
        f"RAGAS evaluator LLM: {args.ragas_llm_model} "
        f"(~{evaluator_rpm:.1f} RPM max)",
        flush=True,
    )
    print(
        f"RAGAS embeddings: {args.ragas_embedding_provider} "
        "(no Gemini embedding requests)",
        flush=True,
    )
    if generation_rpm is not None:
        print(f"Narrative generation pacing: ~{generation_rpm:.1f} RPM max", flush=True)


def _validate_api_key(env_name: str) -> None:
    if not os.environ.get(env_name):
        raise SystemExit(
            f"Missing Gemini API key. Set ${env_name} before running this script."
        )


def _validate_gemini_models(
    api_key_env: str, variant_models: list[tuple[str, str]]
) -> None:
    available = set(_list_gemini_generation_models(api_key_env))
    missing = [
        (method_label, model_id)
        for method_label, model_id in variant_models
        if _normalize_model_id(model_id) not in available
    ]
    if not missing:
        return

    missing_lines = "\n".join(
        f"  - {method_label}: {model_id}" for method_label, model_id in missing
    )
    preview = "\n".join(f"  - {model_id}" for model_id in sorted(available)[:20])
    raise SystemExit(
        "One or more configured Gemini model ids are not available for text "
        "generation with this API key.\n\n"
        f"Configured but unavailable:\n{missing_lines}\n\n"
        "Important: BRANCH-Gemma4-26B/31B are paper table labels; "
        "the API must receive real Gemini model ids such as "
        "gemma-4-26b-a4b-it or gemma-4-31b-it.\n\n"
        f"Available generation models include:\n{preview}\n\n"
        "Run this to see the full list:\n"
        "  python .\\scripts\\run_gemini_variants.py --list-gemini-models"
    )


def _list_gemini_generation_models(api_key_env: str) -> list[str]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing Gemini API key. Set ${api_key_env}.")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = request.Request(endpoint, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"Gemini ListModels failed with HTTP {exc.code}: {details}"
        ) from exc
    except error.URLError as exc:
        raise SystemExit(f"Gemini ListModels request failed: {exc}") from exc

    model_ids = []
    for model in payload.get("models", []):
        methods = set(model.get("supportedGenerationMethods", []))
        if "generateContent" not in methods:
            continue
        name = _normalize_model_id(str(model.get("name", "")))
        if name:
            model_ids.append(name)
    return sorted(set(model_ids))


def _normalize_model_id(model_id: str) -> str:
    return model_id.removeprefix("models/")


def _metrics_output_path(method_label: str, dataset: str) -> Path:
    slug = _variant_slug(method_label)
    return Path("results") / "metrics" / f"explanation_quality_{slug}_{dataset}.csv"


def _variant_artifact_root(base_root: str, method_label: str) -> Path:
    return Path(base_root) / _variant_slug(method_label)


def _variant_slug(method_label: str) -> str:
    return method_label.lower().replace("branch-", "").replace("-", "_")


def _variant_key(method_label: str) -> str:
    if "26b" in method_label.lower():
        return "26b"
    if "31b" in method_label.lower():
        return "31b"
    return method_label.lower()


def _optional_arg(flag: str, value: str | None) -> list[str]:
    return [flag, value] if value else []


def _run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"> {printable}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
