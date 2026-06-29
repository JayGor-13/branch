from dataclasses import dataclass

from branch.evaluation.ragas_external import (
    RagasRecord,
    _record_to_sample,
    _score_sample_with_timeout_fallback,
)


@dataclass
class FakeSample:
    user_input: str
    response: str
    retrieved_contexts: list[str]


class TimeoutThenScoreMetric:
    name = "faithfulness"

    def __init__(self):
        self.calls = 0

    async def single_turn_ascore(self, sample, callbacks=None, timeout=None):
        self.calls += 1
        if len(sample.response) > 1200:
            raise TimeoutError("simulated timeout on large prompt")
        return 0.75


class AnswerMetric:
    name = "answer_relevancy"

    def __init__(self):
        self.calls = 0

    async def single_turn_ascore(self, sample, callbacks=None, timeout=None):
        self.calls += 1
        return 0.5


class AlwaysTimeoutMetric:
    name = "faithfulness"

    async def single_turn_ascore(self, sample, callbacks=None, timeout=None):
        raise TimeoutError("simulated persistent timeout")


def test_record_to_sample_compacts_ragas_payload():
    sample = _record_to_sample(
        FakeSample,
        RagasRecord(
            user_input="q" * 1000,
            response="r" * 4000,
            retrieved_contexts=["c" * 1500, "d" * 1500, "e" * 1500, "f" * 1500],
        ),
    )

    assert len(sample.user_input) <= 700
    assert len(sample.response) <= 2500
    assert len(sample.retrieved_contexts) == 3
    assert all(len(context) <= 900 for context in sample.retrieved_contexts)


def test_timeout_fallback_retries_compacted_sample():
    faithfulness = TimeoutThenScoreMetric()
    answer = AnswerMetric()
    score = _score_sample_with_timeout_fallback(
        sample=FakeSample(
            user_input="Explain patient",
            response="large response " * 300,
            retrieved_contexts=["context " * 300],
        ),
        sample_cls=FakeSample,
        metrics=[faithfulness, answer],
        timeout_sec=1,
        attempts=1,
        retry_wait_sec=0,
    )

    assert score.faithfulness == 0.75
    assert score.answer_relevancy == 0.5
    assert faithfulness.calls == 2
    assert answer.calls == 1


def test_persistent_metric_timeout_keeps_available_metric():
    score = _score_sample_with_timeout_fallback(
        sample=FakeSample(
            user_input="Explain patient",
            response="large response " * 300,
            retrieved_contexts=["context " * 300],
        ),
        sample_cls=FakeSample,
        metrics=[AlwaysTimeoutMetric(), AnswerMetric()],
        timeout_sec=1,
        attempts=1,
        retry_wait_sec=0,
    )

    assert score.faithfulness is None
    assert score.answer_relevancy == 0.5
    assert "faithfulness: TimeoutError" in score.error
