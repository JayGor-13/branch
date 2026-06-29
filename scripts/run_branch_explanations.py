from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.data.dataset_specs import default_model_path, default_processed_dir, normalize_dataset_name
from branch.agents.react_agent import BranchAgent
from branch.agents.narrative_generator import is_valid_narrative


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BRANCH end-to-end explanations.")
    parser.add_argument("--dataset", default="maternal_health")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--llm-provider", default="template")
    parser.add_argument("--llm-model", default="deterministic_template_generator")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default=None)
    parser.add_argument("--llm-timeout-sec", type=int, default=120)
    parser.add_argument("--llm-max-tokens", type=int, default=1600)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument(
        "--llm-request-delay-sec",
        type=float,
        default=0.0,
        help="Pause after each generated LLM narrative to stay under RPM quotas.",
    )
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
    parser.add_argument(
        "--query",
        default="Explain this prediction and include modifiable counterfactuals if needed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip patients whose trace and narrative already exist under artifact-root.",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    agent = BranchAgent.from_artifacts(
        dataset=dataset,
        processed_dir=args.processed_dir or default_processed_dir(dataset),
        model_path=args.model_path or default_model_path(dataset),
        artifact_root=args.artifact_root,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_sec=args.llm_timeout_sec,
        llm_max_tokens=args.llm_max_tokens,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff_sec=args.llm_retry_backoff_sec,
        llm_fallback_to_template=not args.no_llm_fallback,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key_env=args.embedding_api_key_env,
        embedding_timeout_sec=args.embedding_timeout_sec,
        embedding_dimensions=args.embedding_dimensions,
        vector_index_path=args.vector_index_path,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_similarity_threshold=args.retrieval_similarity_threshold,
    )
    patient_ids = agent.test_df["patient_id"].head(args.limit).astype(int).tolist()
    skipped = 0
    saved = 0
    for patient_id in patient_ids:
        if (
            args.skip_existing
            and agent._trace_path(patient_id).exists()
            and agent._narrative_path(patient_id).exists()
            and _is_valid_existing_narrative(agent._narrative_path(patient_id))
        ):
            skipped += 1
            continue
        agent.explain_patient(patient_id, args.query)
        saved += 1
        if args.llm_request_delay_sec > 0:
            sleep(args.llm_request_delay_sec)
    print(
        f"Saved BRANCH traces and narratives for {saved} patients. "
        f"Skipped {skipped} existing patients."
    )


def _is_valid_existing_narrative(path: Path) -> bool:
    try:
        return is_valid_narrative(path.read_text(encoding="utf-8"))
    except OSError:
        return False


if __name__ == "__main__":
    main()
