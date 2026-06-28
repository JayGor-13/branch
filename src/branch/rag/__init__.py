"""Embedding-backed retrieval utilities for BRANCH."""

from branch.rag.embeddings import (
    EmbeddingConfig,
    GeminiEmbeddingClient,
    LocalHashEmbeddingClient,
    build_embedding_client,
    build_embedding_config,
)
from branch.rag.vector_store import (
    GuidelineVectorIndex,
    build_guideline_query,
    build_guideline_vector_index,
    load_or_build_guideline_vector_index,
)

__all__ = [
    "EmbeddingConfig",
    "GeminiEmbeddingClient",
    "GuidelineVectorIndex",
    "LocalHashEmbeddingClient",
    "build_embedding_client",
    "build_embedding_config",
    "build_guideline_query",
    "build_guideline_vector_index",
    "load_or_build_guideline_vector_index",
]
