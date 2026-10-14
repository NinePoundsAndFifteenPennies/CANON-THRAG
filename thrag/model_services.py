"""vLLM chat clients, local Qwen tokenization, and a shared CPU E5 encoder."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from config import THRAGConfig, get_config

E5_DIMENSION = 768


class ChatTextTokenizer:
    """Tokenize raw chat-model text using the configured local tokenizer files."""

    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            if not (self.tokenizer_path / filename).is_file():
                raise FileNotFoundError(f"Missing chat tokenizer file: {self.tokenizer_path / filename}.")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )

    def encode(self, text: str) -> list[int]:
        with self._lock:
            self._load_tokenizer()
            # Chunks contain source text only, without chat BOS/EOS or templates.
            return self._tokenizer.encode(
                text, add_special_tokens=False, truncation=False, verbose=False
            )

    def decode(self, token_ids: list[int]) -> str:
        with self._lock:
            self._load_tokenizer()
            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

    def count_text(self, text: str) -> int:
        """Count raw text; API usage also includes message/template overhead."""
        return len(self.encode(text))

    def count_messages(
        self,
        messages: list[dict[str, str]],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Count the chat template and assistant prefix as part of the input."""
        with self._lock:
            self._load_tokenizer()
            token_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                truncation=False,
                **(chat_template_kwargs or {}),
            )
        return len(token_ids)


_CHAT_TOKENIZERS: dict[str, ChatTextTokenizer] = {}
_CHAT_TOKENIZERS_LOCK = threading.Lock()


def get_chat_tokenizer(tokenizer_path: str | None = None) -> ChatTextTokenizer:
    path = str(Path(tokenizer_path or get_config().chat_tokenizer_path).expanduser().resolve())
    with _CHAT_TOKENIZERS_LOCK:
        if path not in _CHAT_TOKENIZERS:
            _CHAT_TOKENIZERS[path] = ChatTextTokenizer(path)
        return _CHAT_TOKENIZERS[path]


def create_chat_client(config: THRAGConfig | None = None) -> OpenAI:
    config = config or get_config()
    if not config.llm_api_key or config.llm_api_key.startswith("your_"):
        raise ValueError("Set LLM_API_KEY in the server environment or thrag/.env.")
    return OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)


# The nested question/subtopic pools share one limit for the remote LLM service.
# Local E5 encoding keeps its existing independent model lock.
_ANSWER_REQUEST_SLOTS = threading.BoundedSemaphore(get_config().max_workers)


def create_answer_completion(client: OpenAI, **kwargs: Any) -> Any:
    with _ANSWER_REQUEST_SLOTS:
        return client.chat.completions.create(**kwargs)


class CpuE5Encoder:
    """Load local E5 weights and WordPiece files lazily, without CUDA or downloads."""

    def __init__(self, model_path: str, batch_size: int, max_length: int) -> None:
        if batch_size < 1 or not 1 <= max_length <= 512:
            raise ValueError("E5 requires a positive batch size and a maximum length of 1-512.")
        self.model_path = Path(model_path).expanduser().resolve()
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local E5 directory not found: {self.model_path}. Set E5_MODEL_PATH.")

        # Sentence-Transformers loads the model's local WordPiece tokenizer itself.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(self.model_path),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )
        model.max_seq_length = self.max_length
        dimension = model.get_sentence_embedding_dimension()
        if dimension != E5_DIMENSION:
            raise ValueError(f"Expected 768-dimensional e5-base-v2, got {dimension}.")
        self._model = model

    def _encode(self, texts: list[str], prefix: str, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, E5_DIMENSION), dtype="float32")
        inputs = [text if text.startswith(prefix) else prefix + text for text in texts]
        # Query worker threads share one model; serialize CPU encode calls.
        with self._lock:
            self._load_model()
            vectors = self._model.encode(
                inputs,
                batch_size=self.batch_size,
                device="cpu",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.shape != (len(texts), E5_DIMENSION):
            raise ValueError(f"Unexpected E5 output shape: {vectors.shape}.")
        return np.ascontiguousarray(vectors)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, "passage: ", show_progress)


_ENCODERS: dict[tuple[str, int, int], CpuE5Encoder] = {}
_ENCODERS_LOCK = threading.Lock()


def get_cpu_e5_encoder(
    model_path: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
) -> CpuE5Encoder:
    config = get_config()
    path = str(Path(model_path or config.e5_model_path).expanduser().resolve())
    batch_size = config.e5_batch_size if batch_size is None else batch_size
    max_length = config.e5_max_length if max_length is None else max_length
    key = (path, batch_size, max_length)
    with _ENCODERS_LOCK:
        if key not in _ENCODERS:
            _ENCODERS[key] = CpuE5Encoder(path, batch_size, max_length)
        return _ENCODERS[key]
