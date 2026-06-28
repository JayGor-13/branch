from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.guardrails.guideline_loader import (
    default_maternal_guideline_chunks,
    load_guideline_chunks_from_directory,
    save_guideline_chunks,
)
from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.rag.vector_store import build_guideline_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clinical PDF guideline vector index.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset scope for inferred feature directions: all, gallstone, maternal_health, npha, or load_diabetes.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/external/clinical_guidelines/pdfs",
        help="Folder containing clinical guideline PDFs/TXT/MD files.",
    )
    parser.add_argument(
        "--output-path",
        default="data/external/clinical_guidelines/chunks/clinical_guidelines.json",
    )
    parser.add_argument(
        "--vector-index-path",
        default="artifacts/vector_store/clinical_guidelines",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--use-default-curated",
        action="store_true",
        help="Use the small built-in maternal guideline chunks for smoke tests.",
    )
    parser.add_argument("--embedding-provider", default="local")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-api-key-env", default=None)
    parser.add_argument("--embedding-timeout-sec", type=int, default=120)
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    args = parser.parse_args()

    if args.use_default_curated:
        chunks = [chunk.__dict__ for chunk in default_maternal_guideline_chunks()]
    else:
        chunks = load_guideline_chunks_from_directory(
            args.pdf_dir,
            dataset=args.dataset,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    corpus_path = save_guideline_chunks(chunks, args.output_path)
    config = build_embedding_config(
        provider=args.embedding_provider,
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key_env=args.embedding_api_key_env,
        timeout_sec=args.embedding_timeout_sec,
        dimensions=args.embedding_dimensions,
    )
    client = build_embedding_client(config)
    index = build_guideline_vector_index(
        dataset=args.dataset,
        output_dir=args.vector_index_path,
        embedding_client=client,
        chunks=chunks,
    )
    print(f"Saved guideline chunks to {corpus_path}")
    print(
        "Saved guideline vector index to "
        f"{index.index_dir} using {index.manifest['embedding_provider']} "
        f"/ {index.manifest['embedding_model']}"
    )


if __name__ == "__main__":
    main()
