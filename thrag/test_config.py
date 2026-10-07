"""Utility script for validating TH-RAG configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def test_config() -> None:
    print("TH-RAG configuration check")
    print("=" * 60)

    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        if env_example_path.exists():
            print("Missing .env file. Copy .env.example and fill in your API key.")
            print(f"Example: copy {env_example_path} {env_path}")
        else:
            print("Missing both .env and .env.example.")
        return

    try:
        from config import get_config

        config = get_config()
    except Exception as exc:
        print(f"Failed to load configuration: {exc}")
        return

    print("API settings")
    print("-" * 60)
    if config.deepseek_api_key and not config.deepseek_api_key.startswith("your_"):
        print("DEEPSEEK_API_KEY: configured")
    else:
        print("DEEPSEEK_API_KEY: missing or placeholder")
    print(f"DEEPSEEK_BASE_URL: {config.deepseek_base_url}")
    print(f"DEEPSEEK_THINKING: {config.deepseek_thinking}")

    print("\nModel settings")
    print("-" * 60)
    print(f"DEFAULT_MODEL: {config.default_model}")
    print(f"E5_MODEL_PATH: {config.e5_model_path}")
    print("E5 device: cpu")
    print(f"E5_MAX_LENGTH / E5_BATCH_SIZE: {config.e5_max_length} / {config.e5_batch_size}")
    print(f"DEEPSEEK_TOKENIZER_PATH: {config.deepseek_tokenizer_path}")
    print(f"CHAT_MODEL: {config.chat_model}")
    print(f"EVAL_MODEL: {config.eval_model}")

    print("\nRetrieval and generation settings")
    print("-" * 60)
    print(f"TOP_K1 / TOP_K2: {config.top_k1} / {config.top_k2}")
    print(f"TOP_K1_LONG / TOP_K2_LONG: {config.top_k1_long} / {config.top_k2_long}")
    print(f"MAX_TOKENS / OVERLAP: {config.max_tokens} / {config.overlap}")
    print(f"MAX_WORKERS: {config.max_workers}")

    print("\nEnvironment variables")
    print("-" * 60)
    for name in [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_THINKING",
        "DEFAULT_MODEL",
        "E5_MODEL_PATH",
        "E5_MAX_LENGTH",
        "E5_BATCH_SIZE",
        "DEEPSEEK_TOKENIZER_PATH",
        "CHAT_MODEL",
        "EVAL_MODEL",
        "TOP_K1",
        "TOP_K2",
        "MAX_TOKENS",
        "OVERLAP",
    ]:
        value = os.getenv(name)
        if not value:
            print(f"{name}: missing")
        elif "API_KEY" in name:
            print(f"{name}: {'placeholder' if value.startswith('your_') else 'configured'}")
        else:
            print(f"{name}: {value}")



def create_sample_env() -> None:
    template = Path(".env.example")
    if not template.exists():
        raise FileNotFoundError(".env.example is missing.")
    Path(".env").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print("Created .env from .env.example. Update DEEPSEEK_API_KEY before running the pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TH-RAG configuration values.")
    parser.add_argument("--create-env", action="store_true", help="Create .env from .env.example")
    args = parser.parse_args()

    if args.create_env:
        create_sample_env()
    else:
        test_config()
