from branch.guardrails.guideline_loader import load_guideline_chunks_from_directory


def test_guideline_loader_chunks_external_clinical_documents(tmp_path):
    doc_dir = tmp_path / "pdfs"
    doc_dir.mkdir()
    (doc_dir / "maternal_bp_guideline.txt").write_text(
        "Hypertension in pregnancy: elevated systolic blood pressure increases maternal risk.",
        encoding="utf-8",
    )

    chunks = load_guideline_chunks_from_directory(doc_dir, dataset="maternal_health")

    assert chunks
    assert chunks[0]["source_type"] == "clinical_text"
    assert chunks[0]["feature_directions"]["Systolic BP"] == "high_value_increases_risk"
