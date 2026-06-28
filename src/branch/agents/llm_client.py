"""LLM client adapters for BRANCH narrative generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any
from urllib import error, request


@dataclass
class LLMConfig:
    provider: str = "template"
    model_name: str = "deterministic_template_generator"
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str = "BRANCH_LLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_sec: int = 120
    fallback_to_template: bool = True
    max_retries: int = 3
    retry_backoff_sec: float = 2.0


@dataclass
class LLMGenerationResult:
    text: str
    provider: str
    model_name: str
    latency_sec: float | None = None


class OpenAICompatibleLLM:
    """Minimal chat-completions client for OpenAI-compatible servers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def generate(self, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        start = perf_counter()
        body = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.config.timeout_sec) as response:
                    body = response.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if _should_retry_http(exc.code) and attempt < self.config.max_retries:
                    _sleep_before_retry(attempt, self.config.retry_backoff_sec)
                    continue
                raise RuntimeError(
                    f"LLM endpoint returned HTTP {exc.code} at {endpoint}: {details}"
                ) from exc
            except error.URLError as exc:
                if attempt < self.config.max_retries:
                    _sleep_before_retry(attempt, self.config.retry_backoff_sec)
                    continue
                raise RuntimeError(
                    f"LLM endpoint is unavailable at {endpoint}: {exc}"
                ) from exc

        if body is None:
            raise RuntimeError(f"LLM endpoint returned no response body at {endpoint}.")

        parsed = json.loads(body)
        choices = parsed.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response did not include choices: {parsed}")
        message = choices[0].get("message", {})
        text = message.get("content", "").strip()
        if not text:
            raise RuntimeError(f"LLM response did not include text content: {parsed}")
        return LLMGenerationResult(
            text=text,
            provider=self.config.provider,
            model_name=self.config.model_name,
            latency_sec=perf_counter() - start,
        )


def build_llm_config(
    provider: str = "template",
    model_name: str = "deterministic_template_generator",
    base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    timeout_sec: int = 120,
    fallback_to_template: bool = True,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> LLMConfig:
    """Build config with environment overrides for experiment runs."""

    resolved_provider = os.environ.get("BRANCH_LLM_PROVIDER", provider)
    resolved_model = os.environ.get(
        "BRANCH_LLM_MODEL", _default_model_name(resolved_provider, model_name)
    )
    resolved_base_url = os.environ.get(
        "BRANCH_LLM_BASE_URL", base_url or _default_base_url(resolved_provider)
    )
    resolved_api_key_env = os.environ.get(
        "BRANCH_LLM_API_KEY_ENV",
        api_key_env or _default_api_key_env(resolved_provider),
    )

    return LLMConfig(
        provider=resolved_provider,
        model_name=resolved_model,
        base_url=resolved_base_url,
        api_key_env=resolved_api_key_env,
        temperature=float(os.environ.get("BRANCH_LLM_TEMPERATURE", temperature)),
        max_tokens=int(os.environ.get("BRANCH_LLM_MAX_TOKENS", max_tokens)),
        timeout_sec=int(os.environ.get("BRANCH_LLM_TIMEOUT_SEC", timeout_sec)),
        fallback_to_template=_env_bool(
            "BRANCH_LLM_FALLBACK_TO_TEMPLATE", fallback_to_template
        ),
        max_retries=int(os.environ.get("BRANCH_LLM_MAX_RETRIES", max_retries)),
        retry_backoff_sec=float(
            os.environ.get("BRANCH_LLM_RETRY_BACKOFF_SEC", retry_backoff_sec)
        ),
    )


def build_llm_client(config: LLMConfig):
    provider = config.provider.lower()
    if provider in {"template", "none", "deterministic"}:
        return None
    if provider in {
        "openai_compatible",
        "qwen",
        "gewn",
        "ollama",
        "vllm",
        "llama_cpp",
        "gemini",
        "google",
        "gemma",
    }:
        return OpenAICompatibleLLM(config)
    raise ValueError(
        "Unsupported LLM provider. Use 'template', 'openai_compatible', or 'gemini'. "
        f"Received: {config.provider}"
    )


def _default_model_name(provider: str, configured_model: str) -> str:
    if configured_model != "deterministic_template_generator":
        return configured_model
    if provider.lower() in {"gemini", "google", "gemma"}:
        return "gemma-4-31b-it"
    return configured_model


def _default_base_url(provider: str) -> str:
    if provider.lower() in {"gemini", "google", "gemma"}:
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
    return "http://localhost:11434/v1"


def _default_api_key_env(provider: str) -> str:
    if provider.lower() in {"gemini", "google", "gemma"}:
        return "GEMINI_API_KEY"
    return "BRANCH_LLM_API_KEY"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_retry_http(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


def _sleep_before_retry(attempt: int, base_delay: float) -> None:
    sleep(base_delay * (2**attempt))
