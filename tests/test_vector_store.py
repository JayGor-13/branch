from branch.guardrails.guideline_loader import default_maternal_guideline_chunks
from branch.rag.embeddings import build_embedding_client, build_embedding_config
from branch.rag.vector_store import (
    GuidelineVectorIndex,
    build_guideline_query,
    build_guideline_vector_index,
)


def test_guideline_vector_index_persists_and_searches(tmp_path):
    client = build_embedding_client(build_embedding_config(provider="local", dimensions=256))
    index = build_guideline_vector_index(
        dataset="maternal_health",
        output_dir=tmp_path / "index",
        embedding_client=client,
    )

    loaded = GuidelineVectorIndex.load(tmp_path / "index")
    query = build_guideline_query(
        {"predicted_class": "High Risk"},
        {
            "features": [
                {
                    "feature": "BS",
                    "value": 9.0,
                    "shap": 0.4,
                    "direction": "increases_prediction",
                }
            ]
        },
        narrative="The summary discusses elevated blood glucose.",
    )
    results = loaded.search(query, client, top_k=1)

    assert (tmp_path / "index" / "index.faiss").exists()
    assert index.manifest["index_type"] == "faiss_index_flat_ip"
    assert results
    assert results[0]["chunk_id"] == "maternal_glucose_001"


def test_in_memory_vector_index_supports_custom_chunks():
    client = build_embedding_client(build_embedding_config(provider="local", dimensions=128))
    chunks = default_maternal_guideline_chunks()
    index = GuidelineVectorIndex.from_chunks(chunks, client)
    results = index.search("mental health support during pregnancy", client, top_k=1)

    assert results[0]["chunk_id"] == "maternal_mental_health_001"
