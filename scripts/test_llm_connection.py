from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.agents.llm_client import build_llm_client, build_llm_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Test BRANCH LLM API connectivity.")
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--llm-model", default="gemma-4-31b-it")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default=None)
    parser.add_argument("--llm-timeout-sec", type=int, default=30)
    args = parser.parse_args()

    config = build_llm_config(
        provider=args.llm_provider,
        model_name=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout_sec=args.llm_timeout_sec,
        fallback_to_template=False,
    )
    client = build_llm_client(config)
    if client is None:
        raise SystemExit("Provider resolved to template mode; no API call was made.")

    result = client.generate(
        system_prompt="You are a connectivity test. Return exactly one short sentence.",
        user_prompt=(
            "Reply with: BRANCH LLM connection ok. Do not include any extra sections."
        ),
    )
    print("LLM connection succeeded")
    print(f"provider={result.provider}")
    print(f"model={result.model_name}")
    print(f"response={result.text}")


if __name__ == "__main__":
    main()
