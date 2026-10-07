"""DeepSeek chat clients and a shared, offline CPU E5 encoder."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from config import THRAGConfig, get_config

E5_DIMENSION = 768


def create_deepseek_client(config: THRAGConfig | None = None) -> OpenAI:
    config = config or get_config()
    if not config.deepseek_api_key or config.deepseek_api_key.startswith("your_"):
        raise ValueError("Set DEEPSEEK_API_KEY in the server environment or thrag/.env.")
    return OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url)


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
