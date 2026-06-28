"""Persistent FAISS vector store for clinical guideline retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from branch.guardrails.guideline_loader import (
    GuidelineChunk,
    default_maternal_guideline_chunks,
)
from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.utils.io import ensure_dir, read_json, write_json


DEFAULT_VECTOR_INDEX_DIR = Path("artifacts/vector_store/clinical_guidelines")


@dataclass
class GuidelineVectorIndex:
    chunks: list[dict[str, Any]]
    index: Any
    manifest: dict[str, Any]
    index_dir: Path | None = None

    @classmethod
    def from_chunks(
        cls,
        chunks: list[GuidelineChunk | dict[str, Any]],
        embedding_client: Any,
        dataset: str = "maternal_health",
    ) -> "GuidelineVectorIndex":
        faiss = _faiss()
        np = _np()
        chunk_dicts = [_chunk_to_dict(chunk) for chunk in chunks]
        documents = [guideline_chunk_to_text(chunk) for chunk in chunk_dicts]
        vectors = np.asarray(
            embedding_client.embed_documents(documents),
            dtype="float32",
        )
        vectors = _normalize_rows(vectors)
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        manifest = {
            "dataset": dataset,
            "index_type": "faiss_index_flat_ip",
            "similarity": "cosine_via_normalized_inner_product",
            "embedding_provider": getattr(embedding_client, "provider", "unknown"),
            "embedding_model": getattr(embedding_client, "model_name", "unknown"),
            "dimensions": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            "chunk_count": len(chunk_dicts),
        }
        return cls(chunks=chunk_dicts, index=index, manifest=manifest)

    @classmethod
    def load(cls, index_dir: str | Path) -> "GuidelineVectorIndex":
        faiss = _faiss()
        path = Path(index_dir)
        index = faiss.read_index(str(path / "index.faiss"))
        chunks = read_json(path / "chunks.json")
        manifest = read_json(path / "manifest.json")
        return cls(chunks=chunks, index=index, manifest=manifest, index_dir=path)

    def save(self, index_dir: str | Path) -> Path:
        faiss = _faiss()
        path = ensure_dir(index_dir)
        faiss.write_index(self.index, str(path / "index.faiss"))
        write_json(self.chunks, path / "chunks.json")
        write_json(self.manifest, path / "manifest.json")
        stale_numpy_index = path / "embeddings.npy"
        if stale_numpy_index.exists():
            stale_numpy_index.unlink()
        self.index_dir = path
        return path

    def search(
        self,
        query: str,
        embedding_client: Any,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        np = _np()
        query_vector = np.asarray(embedding_client.embed_query(query), dtype="float32")
        query_vector = _normalize_vector(query_vector).reshape(1, -1)
        if self.index.d != query_vector.shape[1]:
            raise RuntimeError(
                "Vector index dimension does not match query embedding dimension. "
                "Rebuild the guideline vector index with the same embedding model."
            )
        max_k = min(max(top_k, 1), len(self.chunks))
        scores, indices = self.index.search(query_vector, max_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            score = float(score)
            if score < similarity_threshold:
                continue
            chunk = dict(self.chunks[index])
            chunk["relevance_score"] = round(score, 4)
            chunk["retrieval_backend"] = "faiss"
            results.append(chunk)
        return results


def build_guideline_vector_index(
    dataset: str = "maternal_health",
    output_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    embedding_client: Any | None = None,
    chunks: list[GuidelineChunk | dict[str, Any]] | None = None,
) -> GuidelineVectorIndex:
    client = embedding_client or build_embedding_client(build_embedding_config())
    index = GuidelineVectorIndex.from_chunks(
        chunks or default_maternal_guideline_chunks(),
        client,
        dataset=dataset,
    )
    index.save(output_dir)
    return index


def load_or_build_guideline_vector_index(
    dataset: str,
    index_dir: str | Path,
    embedding_client: Any,
    chunks: list[GuidelineChunk | dict[str, Any]] | None = None,
) -> GuidelineVectorIndex:
    path = Path(index_dir)
    if _index_files_exist(path):
        index = GuidelineVectorIndex.load(path)
        if _index_matches_client(index, embedding_client):
            return index
    return build_guideline_vector_index(
        dataset=dataset,
        output_dir=path,
        embedding_client=embedding_client,
        chunks=chunks,
    )


def build_guideline_query(
    prediction: dict[str, Any],
    shap_result: dict[str, Any],
    narrative: str | None = None,
) -> str:
    parts = [
        f"Clinical risk prediction: {prediction.get('predicted_class', '')}",
        "Top model drivers and SHAP evidence:",
    ]
    for item in shap_result.get("features", []):
        parts.append(
            "feature={feature}; value={value}; shap={shap}; direction={direction}".format(
                feature=item.get("feature", ""),
                value=item.get("value", ""),
                shap=item.get("shap", ""),
                direction=item.get("direction", ""),
            )
        )
    if narrative:
        parts.extend(["LLM expert narrative to clinically ground:", narrative])
    return "\n".join(parts)


def guideline_chunk_to_text(chunk: dict[str, Any]) -> str:
    directions = ", ".join(
        f"{feature}: {direction}"
        for feature, direction in chunk.get("feature_directions", {}).items()
    )
    keywords = ", ".join(chunk.get("keywords", []))
    return (
        f"title: {chunk.get('topic', 'none')} | "
        f"text: {chunk.get('summary', '')} "
        f"Keywords: {keywords}. "
        f"Clinical feature directions: {directions}. "
        f"Source: {chunk.get('source', '')}."
    )


def _chunk_to_dict(chunk: GuidelineChunk | dict[str, Any]) -> dict[str, Any]:
    return asdict(chunk) if isinstance(chunk, GuidelineChunk) else dict(chunk)


def _index_files_exist(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in ["index.faiss", "chunks.json", "manifest.json"]
    )


def _index_matches_client(index: GuidelineVectorIndex, embedding_client: Any) -> bool:
    manifest = index.manifest
    return (
        manifest.get("embedding_provider") == getattr(embedding_client, "provider", None)
        and manifest.get("embedding_model")
        == getattr(embedding_client, "model_name", None)
    )


def _normalize_rows(matrix: Any):
    np = _np()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: Any):
    np = _np()
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _np():
    import numpy as np

    return np


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Missing FAISS dependency. Install it with "
            "`py -m pip install faiss-cpu` or `py -m pip install -r requirements.txt`."
        ) from exc

    return faiss
