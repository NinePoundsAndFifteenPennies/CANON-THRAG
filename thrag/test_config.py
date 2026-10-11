"""Utility script for validating TH-RAG configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def test_config() -> None:
    print("TH-RAG configuration check")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    env_example_path = project_root / ".env.example"

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

    print("Storage paths")
    print("-" * 60)
    print(f"THRAG_STORAGE_ROOT: {config.storage_root}")
    print(f"Data: {config.data_dir}")
    print(f"Results: {config.results_dir}")
    print(f"Pipeline state: {config.get_pipeline_state_file()}")

    print("\nAPI settings")
    print("-" * 60)
    if config.llm_api_key and not config.llm_api_key.startswith("your_"):
        print("LLM_API_KEY: configured")
    else:
        print("LLM_API_KEY: missing or placeholder")
    print(f"LLM_BASE_URL: {config.llm_base_url}")
    print(f"LLM_THINKING: {config.llm_thinking}")

    print("\nModel settings")
    print("-" * 60)
    print(f"DEFAULT_MODEL: {config.default_model}")
    print(f"E5_MODEL_PATH: {config.e5_model_path}")
    print("E5 device: cpu")
    print(f"E5_MAX_LENGTH / E5_BATCH_SIZE: {config.e5_max_length} / {config.e5_batch_size}")
    print(f"CHAT_TOKENIZER_PATH: {config.chat_tokenizer_path}")
    print(f"Chat tokenizer directory exists: {Path(config.chat_tokenizer_path).is_dir()}")
    print(f"CHAT_MODEL: {config.chat_model}")
    print(f"EVAL_MODEL: {config.eval_model}")

    print("\nRetrieval and generation settings")
    print("-" * 60)
    print(f"TOP_K1 / TOP_K2: {config.top_k1} / {config.top_k2}")
    print(f"TOP_K1_LONG / TOP_K2_LONG: {config.top_k1_long} / {config.top_k2_long}")
    print(f"MAX_TOKENS / OVERLAP: {config.max_tokens} / {config.overlap}")
    print(f"MAX_TOKENS_RESPONSE: {config.max_tokens_response}")
    print(f"MAX_CONTEXT_LENGTH (server metadata): {config.max_context_length}")
    print(f"MAX_WORKERS: {config.max_workers}")

    print("\nEnvironment variables")
    print("-" * 60)
    for name in [
        "THRAG_STORAGE_ROOT",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_THINKING",
        "DEFAULT_MODEL",
        "E5_MODEL_PATH",
        "E5_MAX_LENGTH",
        "E5_BATCH_SIZE",
        "CHAT_TOKENIZER_PATH",
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
    project_root = Path(__file__).resolve().parent
    template = project_root / ".env.example"
    if not template.exists():
        raise FileNotFoundError(".env.example is missing.")
    env_path = project_root / ".env"
    if env_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing settings at {env_path}.")
    env_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print("Created .env from .env.example. Update LLM_API_KEY before running the pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TH-RAG configuration values.")
    parser.add_argument("--create-env", action="store_true", help="Create .env from .env.example")
    args = parser.parse_args()

    if args.create_env:
        create_sample_env()
    else:
        test_config()
