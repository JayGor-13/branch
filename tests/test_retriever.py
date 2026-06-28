from branch.guardrails.retriever import retrieve_guidelines
from branch.rag.embeddings import build_embedding_client, build_embedding_config


def test_vector_retrieval_uses_narrative_document_text(tmp_path):
    embedding_client = build_embedding_client(
        build_embedding_config(provider="local", dimensions=256)
    )
    result = retrieve_guidelines(
        "maternal_health",
        {"predicted_class": "Low Risk"},
        {"features": []},
        narrative="The LLM expert summary mentions gestational diabetes and glucose.",
        top_k=1,
        vector_index_path=tmp_path / "maternal_index",
        embedding_client=embedding_client,
    )
    assert result["retrieved_chunks"]
    assert result["retrieved_chunks"][0]["chunk_id"] == "maternal_diabetes_001"
    assert result["retrieval_backend"] == "faiss"
