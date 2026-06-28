from branch.guardrails.alignment_checker import check_clinical_alignment


def test_high_risk_positive_bp_is_concordant():
    shap_result = {
        "dataset": "maternal_health",
        "patient_id": 17,
        "predicted_class": "High Risk",
        "features": [
            {
                "feature": "Systolic BP",
                "value": 150,
                "shap": 0.2,
                "direction": "increases_prediction",
            }
        ],
    }
    guidelines = {
        "retrieved_chunks": [
            {
                "chunk_id": "maternal_bp_test",
                "source": "curated",
                "summary": "Elevated blood pressure increases concern.",
                "feature_directions": {"Systolic BP": "high_value_increases_risk"},
                "relevance_score": 1.0,
            }
        ]
    }
    result = check_clinical_alignment(shap_result, guidelines)
    assert result["alignment_checks"][0]["alignment"] == "concordant"
    assert result["alignment_checks"][0]["evidence_source"] == "retrieved_guideline_chunks"
    assert result["guardrail_status"] == "no_anomaly"
