"""Embedding clients used by the BRANCH vector retriever."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    model_name: str = "local-hashing-embedding"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    dimensions: int = 768
    timeout_sec: int = 120


class LocalHashEmbeddingClient:
    """Deterministic local embedding client for tests and offline smoke runs."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.config.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.config.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize_vector(vector)


class GeminiEmbeddingClient:
    """Gemini REST embedding client using the native embedContent endpoint."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, task_type="RETRIEVAL_DOCUMENT") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def _embed(self, text: str, task_type: str) -> list[float]:
        model = _normalize_gemini_model_name(self.config.model_name)
        endpoint_model = model.removeprefix("models/")
        endpoint = (
            self.config.base_url.rstrip("/")
            + f"/models/{endpoint_model}:embedContent"
        )
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing Gemini embedding API key in {self.config.api_key_env}."
            )

        content_text = text
        payload: dict[str, Any] = {
            "model": model,
            "content": {"parts": [{"text": content_text}]},
        }
        if "embedding-2" not in model:
            payload["taskType"] = task_type
        elif task_type == "RETRIEVAL_QUERY":
            payload["content"]["parts"][0]["text"] = (
                f"task: search result | query: {text}"
            )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding endpoint returned HTTP {exc.code} at {endpoint}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Embedding endpoint is unavailable at {endpoint}: {exc}"
            ) from exc

        values = _extract_embedding_values(json.loads(body))
        return [float(value) for value in values]


def build_embedding_config(
    provider: str = "local",
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    dimensions: int = 768,
    timeout_sec: int = 120,
) -> EmbeddingConfig:
    resolved_provider = os.environ.get("BRANCH_EMBEDDING_PROVIDER", provider)
    resolved_model = os.environ.get(
        "BRANCH_EMBEDDING_MODEL",
        model_name or _default_model_name(resolved_provider),
    )
    resolved_base_url = os.environ.get(
        "BRANCH_EMBEDDING_BASE_URL",
        base_url or _default_base_url(resolved_provider),
    )
    resolved_api_key_env = os.environ.get(
        "BRANCH_EMBEDDING_API_KEY_ENV",
        api_key_env or _default_api_key_env(resolved_provider),
    )
    resolved_dimensions = int(
        os.environ.get("BRANCH_EMBEDDING_DIMENSIONS", dimensions)
    )
    return EmbeddingConfig(
        provider=resolved_provider,
        model_name=resolved_model,
        base_url=resolved_base_url,
        api_key_env=resolved_api_key_env,
        dimensions=resolved_dimensions,
        timeout_sec=int(os.environ.get("BRANCH_EMBEDDING_TIMEOUT_SEC", timeout_sec)),
    )


def build_embedding_client(config: EmbeddingConfig):
    provider = config.provider.lower()
    if provider in {"local", "hash", "hashing", "deterministic"}:
        return LocalHashEmbeddingClient(config)
    if provider in {"gemini", "google", "gemma"}:
        return GeminiEmbeddingClient(config)
    raise ValueError(
        "Unsupported embedding provider. Use 'local' or 'gemini'. "
        f"Received: {config.provider}"
    )


def _extract_embedding_values(parsed: dict[str, Any]) -> list[float]:
    if "embedding" in parsed and "values" in parsed["embedding"]:
        return parsed["embedding"]["values"]
    embeddings = parsed.get("embeddings", [])
    if embeddings and "values" in embeddings[0]:
        return embeddings[0]["values"]
    raise RuntimeError(f"Embedding response did not include values: {parsed}")


def _default_model_name(provider: str) -> str:
    if provider.lower() in {"gemini", "google", "gemma"}:
        return "gemini-embedding-001"
    return "local-hashing-embedding"


def _default_base_url(provider: str) -> str:
    if provider.lower() in {"gemini", "google", "gemma"}:
        return "https://generativelanguage.googleapis.com/v1beta"
    return "local://branch"


def _default_api_key_env(provider: str) -> str:
    if provider.lower() in {"gemini", "google", "gemma"}:
        return "GEMINI_API_KEY"
    return "BRANCH_EMBEDDING_API_KEY"


def _normalize_gemini_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
