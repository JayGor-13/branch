"""External RAGAS evaluation helpers.

This module is intentionally imported only when real RAGAS evaluation is
requested, so offline/local smoke runs do not require the RAGAS dependency.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class RagasRecord:
    user_input: str
    response: str
    retrieved_contexts: list[str]


@dataclass(frozen=True)
class RagasScore:
    faithfulness: float | None
    answer_relevancy: float | None
    error: str | None = None


def evaluate_ragas_records(
    records: list[RagasRecord],
    llm_model: str = "gemma-4-31b-it",
    embedding_provider: str = "local",
    embedding_model: str = "gemini-embedding-001",
    embedding_dimensions: int = 768,
    api_key_env: str = "GEMINI_API_KEY",
    timeout_sec: int = 300,
    max_workers: int = 1,
    max_retries: int = 1,
    max_wait: int = 60,
    record_delay_sec: float = 0.0,
    llm_min_interval_sec: float = 5.0,
    continue_on_error: bool = True,
    answer_relevancy_strictness: int = 1,
) -> list[RagasScore]:
    """Evaluate records with the real RAGAS package.

    RAGAS uses an evaluator LLM and embeddings. For this project we use Gemini
    through ``langchain-google-genai`` as the evaluator LLM. Embeddings default
    to BRANCH's deterministic local embedding client so Table IV evaluation does
    not burn extra Gemini RPM on RAGAS similarity scoring.
    """

    if not records:
        return []

    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"External RAGAS evaluation requires {api_key_env} or GOOGLE_API_KEY."
        )
    # Pass the key explicitly to LangChain/Google clients. Do not mirror it into
    # GOOGLE_API_KEY, because that can mask the intended GEMINI_API_KEY in shells
    # where both variables are set.
    _install_ragas_vertexai_import_shim()

    try:
        from langchain_core.outputs import Generation, LLMResult
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms.base import BaseRagasLLM
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise ImportError(
            "External RAGAS evaluation requires `ragas` and "
            "`langchain-core`. Install with `python -m pip install -r "
            "requirements.txt`."
        ) from exc

    evaluator_llm = _build_gemini_ragas_llm(
        base_llm_cls=BaseRagasLLM,
        llm_result_cls=LLMResult,
        generation_cls=Generation,
        model=llm_model,
        api_key=api_key,
        timeout_sec=timeout_sec,
        min_interval_sec=llm_min_interval_sec,
    )
    evaluator_embeddings = _build_ragas_embeddings(
        embeddings_wrapper_cls=LangchainEmbeddingsWrapper,
        embedding_provider=embedding_provider,
        embedding_dimensions=embedding_dimensions,
    )
    run_config = RunConfig(
        timeout=timeout_sec,
        max_workers=1,
        max_retries=max_retries,
        max_wait=max_wait,
    )
    metrics = _load_ragas_metrics(
        answer_relevancy_strictness=answer_relevancy_strictness,
        max_retries=max_retries,
    )
    _initialise_metrics(metrics, evaluator_llm, evaluator_embeddings, run_config)

    scores: list[RagasScore] = []
    for index, record in enumerate(records):
        print(f"RAGAS evaluating record {index + 1}/{len(records)}", flush=True)
        sample = _record_to_sample(SingleTurnSample, record)
        try:
            score = _score_sample_with_timeout_fallback(
                sample=sample,
                sample_cls=SingleTurnSample,
                metrics=metrics,
                timeout_sec=timeout_sec,
                attempts=max(1, int(max_retries)),
                retry_wait_sec=min(float(max_wait), max(1.0, record_delay_sec)),
            )
            scores.append(score)
            print(f"RAGAS completed record {index + 1}/{len(records)}", flush=True)
        except Exception as exc:
            if not continue_on_error:
                raise
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"RAGAS failed for record {index + 1}: {error_message}", flush=True)
            scores.append(
                RagasScore(
                    faithfulness=None,
                    answer_relevancy=None,
                    error=error_message,
                )
            )
        if record_delay_sec > 0 and index < len(records) - 1:
            sleep(record_delay_sec)
    return scores


def _record_to_sample(sample_cls, record: RagasRecord):
    return sample_cls(
        user_input=_truncate_text(record.user_input, max_chars=450),
        response=_truncate_text(record.response, max_chars=1600),
        retrieved_contexts=[
            _truncate_text(context, max_chars=650)
            for context in record.retrieved_contexts[:3]
        ],
    )


def _score_sample_with_timeout_fallback(
    sample,
    sample_cls,
    metrics,
    timeout_sec: int,
    attempts: int,
    retry_wait_sec: float,
) -> RagasScore:
    compact_sample = sample_cls(
        user_input=_truncate_text(sample.user_input or "", max_chars=450),
        response=_truncate_text(sample.response or "", max_chars=1200),
        retrieved_contexts=[
            _truncate_text(context, max_chars=500)
            for context in (sample.retrieved_contexts or [])[:2]
        ],
    )
    return _run_async(
        _score_record(
            sample=sample,
            compact_sample=compact_sample,
            metrics=metrics,
            timeout_sec=timeout_sec,
            attempts=attempts,
            retry_wait_sec=retry_wait_sec,
        )
    )


def _load_ragas_metrics(
    answer_relevancy_strictness: int = 1,
    max_retries: int = 1,
) -> list[Any]:
    try:
        from ragas.metrics import AnswerRelevancy, Faithfulness

        return [
            Faithfulness(max_retries=max(1, int(max_retries))),
            AnswerRelevancy(strictness=answer_relevancy_strictness),
        ]
    except ImportError:
        pass

    from ragas.metrics import Faithfulness

    response_relevancy_cls = _import_metric_class(
        ["ResponseRelevancy", "AnswerRelevancy"]
    )
    return [
        Faithfulness(max_retries=max(1, int(max_retries))),
        response_relevancy_cls(strictness=answer_relevancy_strictness),
    ]


def _initialise_metrics(metrics, evaluator_llm, evaluator_embeddings, run_config) -> None:
    for metric in metrics:
        if hasattr(metric, "llm") and getattr(metric, "llm") is None:
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings") and getattr(metric, "embeddings") is None:
            metric.embeddings = evaluator_embeddings
        metric.init(run_config)


async def _score_record(
    sample,
    compact_sample,
    metrics,
    timeout_sec: int,
    attempts: int,
    retry_wait_sec: float,
) -> RagasScore:
    values: dict[str, float | None] = {}
    errors: list[str] = []
    for metric in metrics:
        metric_name = getattr(metric, "name", metric.__class__.__name__)
        print(f"  RAGAS metric started: {metric_name}", flush=True)
        try:
            values[metric_name] = await _score_metric(
                metric=metric,
                sample=sample,
                compact_sample=compact_sample,
                timeout_sec=timeout_sec,
                attempts=attempts,
                retry_wait_sec=retry_wait_sec,
            )
            print(f"  RAGAS metric completed: {metric_name}", flush=True)
        except Exception as exc:
            values[metric_name] = None
            message = f"{metric_name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"  RAGAS metric unavailable: {message}", flush=True)

    return RagasScore(
        faithfulness=_metric_value(values, ["faithfulness"]),
        answer_relevancy=_metric_value(
            values,
            [
                "answer_relevancy",
                "answer_relevance",
                "response_relevancy",
                "response_relevance",
            ],
        ),
        error="; ".join(errors) if errors else None,
    )


async def _score_metric(
    metric,
    sample,
    compact_sample,
    timeout_sec: int,
    attempts: int,
    retry_wait_sec: float,
) -> float:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            value = await metric.single_turn_ascore(
                sample,
                callbacks=[],
                timeout=None,
            )
            return float(value)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                "  RAGAS metric retry "
                f"{attempt + 1}/{attempts} after {type(exc).__name__}: {exc}",
                flush=True,
            )
            await asyncio.sleep(max(0.0, retry_wait_sec))
    assert last_error is not None
    if _is_timeout_error(last_error):
        print(
            "  RAGAS metric timed out; retrying once with compacted sample.",
            flush=True,
        )
        value = await metric.single_turn_ascore(
            compact_sample,
            callbacks=[],
            timeout=None,
        )
        return float(value)
    raise last_error


def _metric_value(values: dict[str, float | None], names: list[str]) -> float | None:
    for name in names:
        value = values.get(name)
        if value is not None:
            return float(value)
    return None


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message or "deadline" in message


def _truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + "\n[truncated]"


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - interactive fallback.
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _build_gemini_ragas_llm(
    base_llm_cls,
    llm_result_cls,
    generation_cls,
    model: str,
    api_key: str,
    timeout_sec: int,
    min_interval_sec: float,
    max_output_tokens: int = 2048,
):
    """Build a minimal RAGAS LLM adapter backed by Gemini generateContent.

    RAGAS only needs a BaseRagasLLM that turns prompt text into generations.
    Calling Gemini directly keeps Table IV independent of LangChain wrapper
    behavior around multiple candidates, retries, and request timeouts.
    """

    class GeminiRagasLLM(base_llm_cls):
        def __init__(self):
            super().__init__()
            self.model = model
            self.model_name = model
            self._api_key = api_key
            self._timeout_sec = int(timeout_sec)
            self._max_output_tokens = int(max_output_tokens)
            self._min_interval_sec = max(0.0, float(min_interval_sec))
            self._rate_lock = threading.Lock()
            self._last_call_started_at = 0.0

        def generate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
            callbacks=None,
        ):
            return self._generate_many(
                prompt,
                n=n,
                temperature=temperature,
                stop=stop,
            )

        async def agenerate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
            callbacks=None,
        ):
            return await asyncio.to_thread(
                self._generate_many,
                prompt,
                n,
                temperature,
                stop,
            )

        def is_finished(self, response) -> bool:
            return True

        def _generate_many(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
        ):
            generations = []
            for _ in range(max(1, int(n))):
                text = self._generate_one(
                    _prompt_to_text(prompt),
                    temperature=0.01 if temperature is None else float(temperature),
                    stop=stop,
                )
                generations.append(generation_cls(text=text))
            return llm_result_cls(generations=[generations])

        def _generate_one(
            self,
            prompt_text: str,
            temperature: float,
            stop: list[str] | None,
        ) -> str:
            self._wait_before_call()
            payload = _gemini_generate_content_payload(
                prompt_text=prompt_text,
                temperature=temperature,
                max_output_tokens=self._max_output_tokens,
                stop=stop,
            )
            response_payload = _post_gemini_generate_content(
                model=self.model,
                api_key=self._api_key,
                payload=payload,
                timeout_sec=self._timeout_sec,
            )
            return _extract_gemini_text(response_payload)

        def _wait_before_call(self) -> None:
            if self._min_interval_sec <= 0:
                return
            with self._rate_lock:
                now = perf_counter()
                wait_sec = self._min_interval_sec - (
                    now - self._last_call_started_at
                )
                if wait_sec > 0:
                    sleep(wait_sec)
                self._last_call_started_at = perf_counter()

    return GeminiRagasLLM()


def _prompt_to_text(prompt) -> str:
    if hasattr(prompt, "text"):
        return str(prompt.text)
    if hasattr(prompt, "to_string"):
        return str(prompt.to_string())
    return str(prompt)


def _gemini_generate_content_payload(
    prompt_text: str,
    temperature: float,
    max_output_tokens: int,
    stop: list[str] | None,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "candidateCount": 1,
        "maxOutputTokens": max_output_tokens,
    }
    if stop:
        generation_config["stopSequences"] = stop
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}],
            }
        ],
        "generationConfig": generation_config,
    }


def _post_gemini_generate_content(
    model: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_sec: int,
) -> dict[str, Any]:
    model_resource = _normalize_gemini_model_resource(model)
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"{model_resource}:generateContent?key={api_key}"
    )
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        guidance = (
            " Increase --ragas-llm-min-interval-sec if this is HTTP 429."
            if exc.code == 429
            else ""
        )
        raise RuntimeError(
            f"Gemini RAGAS evaluator returned HTTP {exc.code}: {details}{guidance}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Gemini RAGAS evaluator request failed: {exc.reason}"
        ) from exc


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini RAGAS evaluator returned no candidates: {payload}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        finish_reason = candidates[0].get("finishReason", "unknown")
        raise RuntimeError(
            "Gemini RAGAS evaluator returned an empty response "
            f"(finishReason={finish_reason})."
        )
    return _strip_json_code_fence(text)


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        import re

        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    return stripped


def _normalize_gemini_model_resource(model: str) -> str:
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _build_ragas_embeddings(
    embeddings_wrapper_cls,
    embedding_provider: str,
    embedding_dimensions: int,
):
    embeddings_wrapper_cls = _resolve_ragas_deprecation_helper(embeddings_wrapper_cls)
    provider = embedding_provider.lower()
    if provider in {"local", "hash", "hashing", "deterministic"}:
        return embeddings_wrapper_cls(_LocalRagasEmbeddings(embedding_dimensions))
    raise ValueError(
        "Unsupported RAGAS embedding provider. Use 'local' for Table IV. "
        f"Received: {embedding_provider}"
    )


def _resolve_ragas_deprecation_helper(obj):
    """Return the real RAGAS class when public imports expose a proxy."""

    return getattr(obj, "new_target", obj)


class _LocalRagasEmbeddings:
    """LangChain-compatible wrapper around BRANCH's local embedding client."""

    def __init__(self, dimensions: int):
        from branch.rag.embeddings import build_embedding_client, build_embedding_config

        config = build_embedding_config(provider="local", dimensions=dimensions)
        self._client = build_embedding_client(config)
        self.model = self._client.model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


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

