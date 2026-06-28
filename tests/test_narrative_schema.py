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


def test_llm_output_is_saved_without_required_sections():
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
    assert "The model suggests high risk" in narrative
    assert "<thought>" not in narrative
