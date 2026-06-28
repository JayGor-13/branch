"""External RAGAS evaluation helpers.

This module is intentionally imported only when real RAGAS evaluation is
requested, so offline/local smoke runs do not require the RAGAS dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import sleep
from typing import Any


@dataclass(frozen=True)
class RagasRecord:
    user_input: str
    response: str
    retrieved_contexts: list[str]


@dataclass(frozen=True)
class RagasScore:
    faithfulness: float | None
    answer_relevancy: float | None


def evaluate_ragas_records(
    records: list[RagasRecord],
    llm_model: str = "gemma-4-31b-it",
    embedding_model: str = "models/text-embedding-004",
    api_key_env: str = "GEMINI_API_KEY",
    max_workers: int = 1,
    max_retries: int = 3,
    max_wait: int = 60,
    record_delay_sec: float = 20.0,
) -> list[RagasScore]:
    """Evaluate records with the real RAGAS package.

    RAGAS uses an evaluator LLM and embeddings. For this project we use Gemini
    through ``langchain-google-genai`` so the evaluation stack matches the paper
    experiments without depending on OpenAI keys.
    """

    if not records:
        return []

    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"External RAGAS evaluation requires {api_key_env} or GOOGLE_API_KEY."
        )
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    _install_ragas_vertexai_import_shim()

    try:
        from datasets import Dataset
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise ImportError(
            "External RAGAS evaluation requires `ragas`, `datasets`, and "
            "`langchain-google-genai`. Install with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=_normalize_embedding_model(embedding_model),
            google_api_key=api_key,
        )
    )
    run_config = RunConfig(
        max_workers=max_workers,
        max_retries=max_retries,
        max_wait=max_wait,
    )

    scores: list[RagasScore] = []
    for index, record in enumerate(records):
        dataset = Dataset.from_dict(_records_to_dataset_dict([record]))
        result = evaluate(
            dataset,
            metrics=_load_ragas_metrics(),
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            raise_exceptions=True,
            batch_size=1,
            show_progress=False,
        )
        frame = result.to_pandas() if hasattr(result, "to_pandas") else result
        scores.extend(
            RagasScore(
                faithfulness=_first_float(row, ["faithfulness"]),
                answer_relevancy=_first_float(
                    row,
                    [
                        "answer_relevancy",
                        "answer_relevance",
                        "response_relevancy",
                        "response_relevance",
                    ],
                ),
            )
            for _, row in frame.iterrows()
        )
        if record_delay_sec > 0 and index < len(records) - 1:
            sleep(record_delay_sec)
    return scores


def _load_ragas_metrics() -> list[Any]:
    try:
        from ragas.metrics import answer_relevancy, faithfulness

        return [faithfulness, answer_relevancy]
    except ImportError:
        pass

    from ragas.metrics import Faithfulness

    response_relevancy_cls = _import_metric_class(
        ["ResponseRelevancy", "AnswerRelevancy"]
    )
    return [Faithfulness(), response_relevancy_cls()]


def _install_ragas_vertexai_import_shim() -> None:
    """Work around a RAGAS/LangChain import mismatch.

    RAGAS 0.4.x imports ``langchain_community.chat_models.vertexai`` while recent
    LangChain Community releases removed that chat-model module. We do not use
    Vertex AI for BRANCH evaluation, but the import still happens during RAGAS
    initialization. Supplying a minimal module keeps RAGAS importable without
    changing its installed package files.
    """

    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim only.
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def _import_metric_class(names: list[str]):
    import ragas.metrics as metrics

    for name in names:
        if hasattr(metrics, name):
            return getattr(metrics, name)
    raise ImportError(
        "Could not import a RAGAS answer relevancy metric class. Tried: "
        + ", ".join(names)
    )


def _records_to_dataset_dict(records: list[RagasRecord]) -> dict[str, list[Any]]:
    user_inputs = [record.user_input for record in records]
    responses = [record.response for record in records]
    contexts = [record.retrieved_contexts for record in records]
    return {
        # Newer RAGAS sample schema.
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": contexts,
        # Older RAGAS dataset schema.
        "question": user_inputs,
        "answer": responses,
        "contexts": contexts,
    }


def _normalize_embedding_model(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def _first_float(row, columns: list[str]) -> float | None:
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if value is None or str(value).strip() == "":
            continue
        return float(value)
    return None
