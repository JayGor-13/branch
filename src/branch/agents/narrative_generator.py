"""Faithful narrative generation from structured BRANCH tool outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from branch.agents.prompts import NARRATIVE_RULES


@dataclass
class NarrativeGenerationResult:
    text: str
    used_llm: bool
    fallback_used: bool
    provider: str
    model_name: str
    fallback_reason: str | None = None
    llm_latency_sec: float | None = None


def generate_narrative(
    query: str,
    prediction: dict[str, Any],
    shap_result: dict[str, Any],
    dice_result: dict[str, Any] | None,
    guideline_context: dict[str, Any],
    guardrail_result: dict[str, Any],
    llm_client: Any | None = None,
    fallback_to_template: bool = True,
) -> str:
    return generate_narrative_result(
        query,
        prediction,
        shap_result,
        dice_result,
        guideline_context,
        guardrail_result,
        llm_client=llm_client,
        fallback_to_template=fallback_to_template,
    ).text


def generate_narrative_result(
    query: str,
    prediction: dict[str, Any],
    shap_result: dict[str, Any],
    dice_result: dict[str, Any] | None,
    guideline_context: dict[str, Any],
    guardrail_result: dict[str, Any],
    llm_client: Any | None = None,
    fallback_to_template: bool = True,
) -> NarrativeGenerationResult:
    if llm_client is not None:
        try:
            system_prompt, user_prompt = build_llm_prompts(
                query,
                prediction,
                shap_result,
                dice_result,
                guideline_context,
                guardrail_result,
            )
            result = llm_client.generate(system_prompt, user_prompt)
            return NarrativeGenerationResult(
                text=_prepare_llm_output(result.text),
                used_llm=True,
                fallback_used=False,
                provider=result.provider,
                model_name=result.model_name,
                llm_latency_sec=result.latency_sec,
            )
        except Exception as exc:
            if not fallback_to_template:
                raise
            template = generate_template_narrative(
                query,
                prediction,
                shap_result,
                dice_result,
                guideline_context,
                guardrail_result,
            )
            reason = str(exc).replace("--", "-")
            return NarrativeGenerationResult(
                text=template + f"\n\n<!-- LLM fallback used: {reason} -->",
                used_llm=False,
                fallback_used=True,
                provider=getattr(llm_client, "provider", "unknown"),
                model_name=getattr(llm_client, "model_name", "unknown"),
                fallback_reason=reason,
            )

    return NarrativeGenerationResult(
        text=generate_template_narrative(
            query,
            prediction,
            shap_result,
            dice_result,
            guideline_context,
            guardrail_result,
        ),
        used_llm=False,
        fallback_used=False,
        provider="template",
        model_name="deterministic_template_generator",
    )


def generate_template_narrative(
    query: str,
    prediction: dict[str, Any],
    shap_result: dict[str, Any],
    dice_result: dict[str, Any] | None,
    guideline_context: dict[str, Any],
    guardrail_result: dict[str, Any],
) -> str:
    patient_id = prediction.get("patient_id", shap_result.get("patient_id"))
    probability = prediction.get("predicted_probability", 0.0)
    predicted_value = prediction.get("predicted_value")
    pred_class = prediction.get("predicted_class", "unknown")
    features = shap_result.get("features", [])
    upward = [item for item in features if item["direction"] == "increases_prediction"]
    downward = [item for item in features if item["direction"] == "decreases_prediction"]

    lines = [
        f"# BRANCH Explanation: Patient {patient_id}",
        "",
        "## Prediction",
        _prediction_sentence(pred_class, probability, predicted_value),
        "",
        "## Main Model Drivers",
        _feature_sentence(
            "The largest contributors toward the predicted class were", upward
        ),
        _feature_sentence(
            "The largest contributors away from the predicted class were", downward
        ),
        "",
        "## Counterfactual Pathway",
        _counterfactual_sentence(dice_result),
        "",
        "## Clinical Evidence Retrieved",
        _guideline_sentence(guideline_context),
        "",
        "## Guardrail Status",
        _guardrail_sentence(guardrail_result),
        "",
        "## Caution",
        (
            "This is a model explanation, not a medical diagnosis. Clinician "
            "review is required."
        ),
        "",
        "<!-- Query: " + query.replace("--", "-") + " -->",
    ]
    return "\n".join(lines)


def build_llm_prompts(
    query: str,
    prediction: dict[str, Any],
    shap_result: dict[str, Any],
    dice_result: dict[str, Any] | None,
    guideline_context: dict[str, Any],
    guardrail_result: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are the BRANCH clinical explainability narrator. Generate a doctor-friendly "
        "summary from model evidence only: prediction, probability, SHAP drivers, "
        "and DiCE counterfactuals. Do not diagnose. Do not recommend treatment. "
        "Use retrieved clinical guideline snippets only as grounding context. "
        "Use cautious language such as 'the model suggests' and 'may indicate'. "
        "Return only the explanation text."
    )
    payload = {
        "clinician_query": query,
        "rules": NARRATIVE_RULES,
        "required_sections": [
            "## Prediction",
            "## Main Model Drivers",
            "## Model Evidence Interpretation",
            "## Counterfactual Pathway",
            "## Clinical Evidence Retrieved",
            "## Guardrail Status",
            "## Caution",
        ],
        "prediction": prediction,
        "shap_result": {
            "patient_id": shap_result.get("patient_id"),
            "predicted_class": shap_result.get("predicted_class"),
            "features": shap_result.get("features", []),
        },
        "dice_result": dice_result,
        "guideline_context": guideline_context,
        "guardrail_result": guardrail_result,
        "note": "Do not invent claims beyond the provided model and retrieved guideline evidence.",
    }
    user_prompt = (
        "Write the BRANCH expert summary for this patient using the model evidence "
        "below. This summary will later be evaluated by RAG and EQS. You may use "
        "plain paragraphs or Markdown. Do not include hidden reasoning or code fences.\n\n"
        "# BRANCH Explanation: Patient <patient_id>\n\n"
        "## Prediction\n"
        "<one short paragraph>\n\n"
        "## Main Model Drivers\n"
        "<one short paragraph; mention only SHAP features listed in JSON>\n\n"
        "## Model Evidence Interpretation\n"
        "<one short paragraph explaining what the model appears to rely on>\n\n"
        "## Counterfactual Pathway\n"
        "<one short paragraph>\n\n"
        "## Clinical Evidence Retrieved\n"
        "<one short paragraph; summarize only retrieved guideline chunks>\n\n"
        "## Guardrail Status\n"
        "<one short paragraph with anomaly status>\n\n"
        "## Caution\n"
        "This is a model explanation, not a medical diagnosis. Clinician review is required.\n\n"
        "Structured BRANCH tool outputs:\n\n"
        + json.dumps(payload, indent=2)
    )
    return system_prompt, user_prompt


def _ensure_required_sections(text: str) -> str:
    text = _normalize_llm_markdown(text)
    required = [
        "## Prediction",
        "## Main Model Drivers",
        "## Model Evidence Interpretation",
        "## Counterfactual Pathway",
        "## Caution",
    ]
    missing = [section for section in required if section not in text]
    if missing:
        preview = text[:1000].replace("\n", "\\n")
        raise RuntimeError(
            f"LLM narrative missing required sections: {missing}. "
            f"LLM output preview: {preview}"
        )
    return text


def _prepare_llm_output(text: str) -> str:
    cleaned = _strip_thought_blocks(text).strip()
    if not cleaned:
        cleaned = text.strip()
    return _normalize_llm_markdown(cleaned)


def _strip_thought_blocks(text: str) -> str:
    return re.sub(r"(?is)<thought>.*?</thought>", "", text).strip()


def _normalize_llm_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    section_names = [
        "Prediction",
        "Main Model Drivers",
        "Model Evidence Interpretation",
        "Counterfactual Pathway",
        "Clinical Evidence Retrieved",
        "Guardrail Status",
        "Caution",
    ]
    for name in section_names:
        patterns = [
            rf"(?im)^\s*#{1,6}\s*{re.escape(name)}\s*:?\s*$",
            rf"(?im)^\s*\*\*{re.escape(name)}\*\*\s*:?\s*$",
            rf"(?im)^\s*{re.escape(name)}\s*:\s*$",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, f"## {name}", cleaned)

    if not cleaned.startswith("# BRANCH Explanation"):
        cleaned = "# BRANCH Explanation\n\n" + cleaned
    return cleaned


def _feature_sentence(prefix: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return f"{prefix}: none among the top SHAP features."
    chunks = [
        f"{item['feature']} (value {item['value']}, SHAP {item['shap']:.4f})"
        for item in items[:3]
    ]
    return f"{prefix}: {', '.join(chunks)}."


def _counterfactual_sentence(dice_result: dict[str, Any] | None) -> str:
    if not dice_result:
        return "Counterfactual generation was not requested for this patient."
    validity = dice_result.get("validity")
    if validity in {"not_required", "not_generated", "dependency_missing"}:
        return dice_result.get("notes", "No valid counterfactual was generated.")
    changes = dice_result.get("changes", [])
    if not changes:
        return "No feature changes were required by the counterfactual tool."
    change_text = ", ".join(
        f"{item['feature']} from {item['from']} to {item['to']}" for item in changes
    )
    return (
        "The model would move toward "
        f"{dice_result.get('counterfactual_prediction', 'a lower-risk class')} if "
        f"{change_text}, assuming other features remain fixed."
    )


def _prediction_sentence(
    pred_class: str,
    probability: float | None,
    predicted_value: float | None,
) -> str:
    if predicted_value is not None:
        return f"The model predicts a continuous outcome value of {predicted_value:.3f}."
    probability = probability or 0.0
    return f"The model predicts {pred_class} with {probability * 100:.1f}% confidence."


def _guideline_sentence(guideline_context: dict[str, Any]) -> str:
    chunks = guideline_context.get("retrieved_chunks", [])
    if not chunks:
        return "No clinical guideline chunks were retrieved for this explanation."
    rendered = []
    for chunk in chunks[:3]:
        topic = chunk.get("topic", "retrieved clinical evidence")
        source = chunk.get("source", "unknown source")
        rendered.append(f"{topic} from {source}")
    return "Retrieved evidence used for grounding: " + "; ".join(rendered) + "."


def _guardrail_sentence(guardrail_result: dict[str, Any]) -> str:
    status = guardrail_result.get("guardrail_status", "unknown")
    warning = guardrail_result.get("warning")
    if warning:
        return f"Guardrail status: {status}. {warning}"
    return f"Guardrail status: {status}."
