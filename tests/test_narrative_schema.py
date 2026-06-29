from branch.agents.narrative_generator import generate_narrative
from branch.agents.narrative_generator import _ensure_required_sections


def test_narrative_contains_required_sections():
    narrative = generate_narrative(
        query="Explain patient 1",
        prediction={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "predicted_probability": 0.82,
        },
        shap_result={
            "features": [
                {
                    "feature": "Systolic BP",
                    "value": 150,
                    "shap": 0.2,
                    "direction": "increases_prediction",
                }
            ]
        },
        dice_result=None,
        guideline_context={
            "retrieved_chunks": [
                {"summary": "Elevated blood pressure is associated with risk."}
            ]
        },
        guardrail_result={"guardrail_status": "no_anomaly"},
    )
    assert "## Prediction" in narrative
    assert "## Main Model Drivers" in narrative
    assert "## Counterfactual Pathway" in narrative
    assert "Systolic BP" in narrative


class FakeLLM:
    provider = "openai_compatible"
    model_name = "fake-qwen"

    def generate(self, system_prompt, user_prompt):
        from branch.agents.llm_client import LLMGenerationResult

        return LLMGenerationResult(
            text=(
                "# BRANCH Explanation: Patient 1\n\n"
                "## Prediction\nThe model suggests High Risk.\n\n"
                "## Main Model Drivers\nSystolic BP was listed by SHAP.\n\n"
                "## Model Evidence Interpretation\nThe summary is grounded in model outputs.\n\n"
                "## Counterfactual Pathway\nNo pathway was requested.\n\n"
                "## Clinical Evidence Retrieved\nNo guideline chunks were retrieved.\n\n"
                "## Guardrail Status\nNo anomaly was flagged.\n\n"
                "## Caution\nClinician review is required."
            ),
            provider=self.provider,
            model_name=self.model_name,
        )


def test_narrative_can_use_llm_client():
    narrative = generate_narrative(
        query="Explain patient 1",
        prediction={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "predicted_probability": 0.82,
        },
        shap_result={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "features": [
                {
                    "feature": "Systolic BP",
                    "value": 150,
                    "shap": 0.2,
                    "direction": "increases_prediction",
                }
            ],
        },
        dice_result=None,
        guideline_context={"retrieved_chunks": []},
        guardrail_result={"guardrail_status": "no_anomaly"},
        llm_client=FakeLLM(),
    )
    assert "The model suggests High Risk" in narrative


def test_llm_markdown_normalizes_common_section_labels():
    raw = """Prediction:
The model suggests High Risk.

Main Model Drivers:
Systolic BP was listed by SHAP.

Model Evidence Interpretation:
The summary is grounded in model outputs.

Counterfactual Pathway:
No pathway was requested.

Clinical Evidence Retrieved:
No guideline chunks were retrieved.

Guardrail Status:
No anomaly was flagged.

Caution:
Clinician review is required.
"""
    normalized = _ensure_required_sections(raw)
    assert "## Prediction" in normalized
    assert "## Main Model Drivers" in normalized
    assert "## Caution" in normalized


class LooseLLM:
    provider = "gemini"
    model_name = "gemma-test"

    def generate(self, system_prompt, user_prompt):
        from branch.agents.llm_client import LLMGenerationResult

        return LLMGenerationResult(
            text=(
                "<thought>I should plan the answer.</thought>\n"
                "The model suggests high risk because BS and Systolic BP were "
                "important model drivers."
            ),
            provider=self.provider,
            model_name=self.model_name,
        )


def test_llm_output_without_required_sections_is_repaired_when_fallback_disabled():
    narrative = generate_narrative(
        query="Explain patient 1",
        prediction={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "predicted_probability": 0.82,
        },
        shap_result={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "features": [
                {
                    "feature": "BS",
                    "value": 9.0,
                    "shap": 0.2,
                    "direction": "increases_prediction",
                }
            ],
        },
        dice_result=None,
        guideline_context={"retrieved_chunks": []},
        guardrail_result={"guardrail_status": "no_anomaly"},
        llm_client=LooseLLM(),
        fallback_to_template=False,
    )

    assert "## Prediction" in narrative
    assert "## Caution" in narrative
    assert "<thought>" not in narrative


class MissingCautionLLM:
    provider = "gemini"
    model_name = "gemma-test"

    def generate(self, system_prompt, user_prompt):
        from branch.agents.llm_client import LLMGenerationResult

        return LLMGenerationResult(
            text=(
                "# BRANCH Explanation: Patient 1\n\n"
                "## Prediction\nThe model predicts High Risk.\n\n"
                "## Main Model Drivers\nThe LLM-specific driver paragraph is retained.\n\n"
                "## Model Evidence Interpretation\nThe model relies on BS.\n\n"
                "## Counterfactual Pathway\nNo pathway was requested.\n\n"
                "## Clinical Evidence Retrieved\nNo guideline chunks were retrieved.\n\n"
                "## Guardrail Status\nNo anomaly was flagged."
            ),
            provider=self.provider,
            model_name=self.model_name,
        )


def test_partial_llm_output_repairs_missing_caution_without_losing_llm_text():
    narrative = generate_narrative(
        query="Explain patient 1",
        prediction={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "predicted_probability": 0.82,
        },
        shap_result={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "features": [
                {
                    "feature": "BS",
                    "value": 9.0,
                    "shap": 0.2,
                    "direction": "increases_prediction",
                }
            ],
        },
        dice_result=None,
        guideline_context={"retrieved_chunks": []},
        guardrail_result={"guardrail_status": "no_anomaly"},
        llm_client=MissingCautionLLM(),
        fallback_to_template=False,
    )

    assert "The LLM-specific driver paragraph is retained." in narrative
    assert "## Caution" in narrative
    assert "Clinician review is required" in narrative


class BulletHeadingMissingCautionLLM:
    provider = "gemini"
    model_name = "gemma-test"

    def generate(self, system_prompt, user_prompt):
        from branch.agents.llm_client import LLMGenerationResult

        return LLMGenerationResult(
            text=(
                "# BRANCH Explanation\n\n"
                "*   **## Prediction**\nThe model predicts No Gallstone.\n\n"
                "*   **## Main Model Drivers**\n"
                "The bullet-form LLM driver paragraph is retained.\n\n"
                "*   **## Model Evidence Interpretation**\nThe model relies on CRP.\n\n"
                "*   **## Counterfactual Pathway**\nNo pathway was needed.\n\n"
                "*   **## Clinical Evidence Retrieved**\nGallstone evidence was retrieved.\n\n"
                "*   **## Guardrail Status**\nNo anomaly was flagged."
            ),
            provider=self.provider,
            model_name=self.model_name,
        )


def test_bullet_heading_llm_output_is_normalized_and_repaired():
    narrative = generate_narrative(
        query="Explain patient 1",
        prediction={
            "patient_id": 1,
            "predicted_class": "No Gallstone",
            "predicted_probability": 0.71,
        },
        shap_result={"patient_id": 1, "predicted_class": "No Gallstone", "features": []},
        dice_result=None,
        guideline_context={"retrieved_chunks": []},
        guardrail_result={"guardrail_status": "no_anomaly"},
        llm_client=BulletHeadingMissingCautionLLM(),
        fallback_to_template=False,
    )

    assert "*   **##" not in narrative
    assert "The bullet-form LLM driver paragraph is retained." in narrative
    assert "## Caution" in narrative


def test_llm_output_without_required_sections_uses_template_fallback():
    narrative = generate_narrative(
        query="Explain patient 1",
        prediction={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "predicted_probability": 0.82,
        },
        shap_result={
            "patient_id": 1,
            "predicted_class": "High Risk",
            "features": [
                {
                    "feature": "BS",
                    "value": 9.0,
                    "shap": 0.2,
                    "direction": "increases_prediction",
                }
            ],
        },
        dice_result=None,
        guideline_context={"retrieved_chunks": []},
        guardrail_result={"guardrail_status": "no_anomaly"},
        llm_client=LooseLLM(),
        fallback_to_template=True,
    )
    assert "## Prediction" in narrative
    assert "<thought>" not in narrative
