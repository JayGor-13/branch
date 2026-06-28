from branch.evaluation.llm_judge import judge_clinical_alignment


class FakeJudgeLLM:
    model_name = "fake-expert-judge"

    def generate(self, system_prompt, user_prompt):
        from branch.agents.llm_client import LLMGenerationResult

        return LLMGenerationResult(
            text='{"score": 1.0, "label": "aligned", "rationale": "Guidelines support the cited drivers."}',
            provider="gemini",
            model_name=self.model_name,
        )


class RepairJudgeLLM:
    model_name = "fake-repair-judge"

    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        from branch.agents.llm_client import LLMGenerationResult

        self.calls += 1
        if self.calls == 1:
            text = "<thought>The explanation seems partly aligned but incomplete.</thought>"
        else:
            text = '{"score": 0.5, "label": "partial_or_unclear", "rationale": "The first response was unclear."}'
        return LLMGenerationResult(
            text=text,
            provider="gemini",
            model_name=self.model_name,
        )


def test_expert_llm_judge_returns_alignment_score():
    result = judge_clinical_alignment(
        FakeJudgeLLM(),
        "The model suggests high risk because BS is elevated.",
        {
            "prediction": {"predicted_class": "High Risk"},
            "guideline_context": {"retrieved_chunks": [{"summary": "Elevated glucose increases risk."}]},
            "guardrail_result": {"alignment_checks": []},
        },
        {"features": [{"feature": "BS", "shap": 0.2}]},
    )
    assert result.score == 1.0
    assert result.used_llm is True


def test_expert_llm_judge_repairs_non_json_output():
    client = RepairJudgeLLM()
    result = judge_clinical_alignment(
        client,
        "The model suggests high risk because BS is elevated.",
        {
            "prediction": {"predicted_class": "High Risk"},
            "guideline_context": {"retrieved_chunks": [{"summary": "Elevated glucose increases risk."}]},
            "guardrail_result": {"alignment_checks": []},
        },
        {"features": [{"feature": "BS", "shap": 0.2}]},
    )
    assert client.calls == 2
    assert result.score == 0.5
