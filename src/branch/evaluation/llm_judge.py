"""Expert LLM judge for explanation clinical alignment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AlignmentJudgeResult:
    score: float
    label: str
    rationale: str
    used_llm: bool
    model_name: str
    raw_response: str | None = None


def judge_clinical_alignment(
    llm_client: Any,
    narrative: str,
    trace: dict[str, Any],
    shap_result: dict[str, Any],
) -> AlignmentJudgeResult:
    system_prompt = (
        "You are an expert clinical explainability evaluator. Judge whether the "
        "retrieved LLM narrative is faithful to the SHAP feature directions, "
        "clinically plausible, and not misleading. Return JSON only."
    )
    payload = {
        "rubric": {
            "1.0": "Faithful to SHAP evidence, clinically plausible, and no serious unsupported clinical claims.",
            "0.5": "Partially aligned, unclear, incomplete, or insufficiently grounded in the model evidence.",
            "0.0": "Clinically discordant, misleading, or contradicts the model/SHAP evidence.",
        },
        "allowed_scores": [1.0, 0.5, 0.0],
        "required_json_schema": {
            "score": "one of 1.0, 0.5, 0.0",
            "label": "aligned | partial_or_unclear | discordant",
            "rationale": "short reason grounded in evidence",
        },
        "narrative": narrative,
        "prediction": trace.get("prediction"),
        "top_shap_features": shap_result.get("features", []),
        "retrieved_guidelines": trace.get("guideline_context", {}).get(
            "retrieved_chunks", []
        ),
        "retrieved_narratives": trace.get("narrative_rag_context", {}).get(
            "retrieved_narratives", []
        ),
        "selected_retrieved_narrative": trace.get("narrative_rag_context", {}).get(
            "selected_narrative"
        ),
        "guardrail_alignment_checks": trace.get("guardrail_result", {}).get(
            "alignment_checks", []
        ),
    }
    user_prompt = (
        "Return only JSON matching the schema. Do not include markdown, code fences, "
        "or hidden reasoning.\n\n"
        + json.dumps(payload, indent=2)
    )
    response = llm_client.generate(system_prompt, user_prompt)
    try:
        parsed = _parse_json_object(response.text)
        raw_response = response.text
    except ValueError:
        repair = llm_client.generate(
            system_prompt=(
                "Convert the evaluator response into strict JSON only. Do not add "
                "markdown, prose, or hidden reasoning."
            ),
            user_prompt=(
                "Convert this response into JSON with keys score, label, and rationale. "
                "Allowed score values are 1.0, 0.5, and 0.0. If the response is unclear, "
                "use score 0.5 and label partial_or_unclear.\n\n"
                + response.text
            ),
        )
        parsed = _parse_json_object(repair.text)
        raw_response = response.text + "\n\n--- repair response ---\n\n" + repair.text
    score = _normalize_score(parsed.get("score"))
    label = str(parsed.get("label", _label_for_score(score)))
    rationale = str(parsed.get("rationale", "No rationale provided."))
    return AlignmentJudgeResult(
        score=score,
        label=label,
        rationale=rationale,
        used_llm=True,
        model_name=response.model_name,
        raw_response=raw_response,
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"(?is)<thought>.*?</thought>", "", text).strip()
    if not cleaned:
        cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return JSON: {text[:500]}")
        return json.loads(match.group(0))


def _normalize_score(value: Any) -> float:
    score = float(value)
    if score >= 0.75:
        return 1.0
    if score >= 0.25:
        return 0.5
    return 0.0


def _label_for_score(score: float) -> str:
    if score == 1.0:
        return "aligned"
    if score == 0.5:
        return "partial_or_unclear"
    return "discordant"
