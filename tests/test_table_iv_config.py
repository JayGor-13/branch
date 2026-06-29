from types import SimpleNamespace

import pytest

from scripts.evaluate_explanations import _validate_ragas_config
from scripts.run_gemini_variants import _validate_table_iv_config


def test_evaluate_ragas_config_requires_local_embeddings():
    args = SimpleNamespace(
        quality_mode="ragas",
        limit=1,
        ragas_embedding_provider="gemini",
        ragas_llm_min_interval_sec=5.0,
    )

    with pytest.raises(SystemExit):
        _validate_ragas_config(args)


def test_evaluate_ragas_config_enforces_15_rpm_limit():
    args = SimpleNamespace(
        quality_mode="ragas",
        limit=1,
        ragas_embedding_provider="local",
        ragas_llm_min_interval_sec=1.0,
        ragas_api_key_env="GEMINI_API_KEY",
    )

    with pytest.raises(SystemExit):
        _validate_ragas_config(args)


def test_evaluate_ragas_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    args = SimpleNamespace(
        quality_mode="ragas",
        limit=1,
        ragas_embedding_provider="local",
        ragas_llm_min_interval_sec=5.0,
        ragas_api_key_env="GEMINI_API_KEY",
    )

    with pytest.raises(SystemExit):
        _validate_ragas_config(args)


def test_run_gemini_variants_config_accepts_table_iv_defaults():
    args = SimpleNamespace(
        quality_mode="ragas",
        limit=10,
        ragas_embedding_provider="local",
        ragas_llm_min_interval_sec=5.0,
        llm_provider="gemini",
        llm_request_delay_sec=5.0,
    )

    _validate_table_iv_config(args, [("BRANCH-Gemma4-26B", "gemma-4-26b-a4b-it")])


def test_run_gemini_variants_config_caps_patients():
    args = SimpleNamespace(
        quality_mode="ragas",
        limit=11,
        ragas_embedding_provider="local",
        ragas_llm_min_interval_sec=5.0,
        llm_provider="gemini",
        llm_request_delay_sec=5.0,
    )

    with pytest.raises(SystemExit):
        _validate_table_iv_config(args, [("BRANCH-Gemma4-26B", "gemma-4-26b-a4b-it")])
