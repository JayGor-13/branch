"""Embedding-backed retrieval over clinical guideline chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.rag.vector_store import (
    DEFAULT_VECTOR_INDEX_DIR,
    GuidelineVectorIndex,
    build_guideline_query,
    load_or_build_guideline_vector_index,
)


def retrieve_guidelines(
    dataset: str,
    prediction: dict[str, Any],
    shap_result: dict[str, Any],
    narrative: str | None = None,
    top_k: int = 3,
    similarity_threshold: float = 0.0,
    vector_index_path: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    embedding_client: Any | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    client = embedding_client or build_embedding_client(build_embedding_config())
    if chunks is None:
        index = load_or_build_guideline_vector_index(
            dataset=dataset,
            index_dir=vector_index_path,
            embedding_client=client,
        )
    else:
        index = GuidelineVectorIndex.from_chunks(chunks, client, dataset=dataset)
    query = build_guideline_query(prediction, shap_result, narrative)
    retrieved = index.search(
        query,
        embedding_client=client,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    return {
        "dataset": dataset,
        "prediction": prediction.get("predicted_class"),
        "retrieval_backend": "faiss",
        "vector_index_path": str(vector_index_path),
        "embedding_provider": getattr(client, "provider", "unknown"),
        "embedding_model": getattr(client, "model_name", "unknown"),
        "query_preview": query[:500],
        "retrieved_chunks": retrieved,
    }
