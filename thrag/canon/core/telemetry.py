"""Per-query client-call accounting, including existing selector helpers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterator

_ACTIVE: ContextVar[tuple[list[dict[str, Any]], str] | None] = ContextVar("canon_calls", default=None)


@contextmanager
def record_calls(calls: list[dict[str, Any]], stage: str) -> Iterator[None]:
    token = _ACTIVE.set((calls, stage))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


class MeteredClient:
    """Wrap only the used SDK surface; no change to existing model helpers."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kwargs: Any) -> MeteredClient:
        return MeteredClient(self._client.with_options(**kwargs))

    def _create(self, **kwargs: Any) -> Any:
        active = _ACTIVE.get()
        started = perf_counter()
        entry: dict[str, Any] = {
            "stage": active[1] if active else "unknown", "model": kwargs.get("model"),
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
        }
        try:
            response = self._client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                entry[key] = getattr(usage, key, None)
            entry["status"] = "ok"
            return response
        except Exception as exc:
            # Never persist exception bodies that may contain request headers/credentials.
            entry.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            entry["seconds"] = perf_counter() - started
            if active is not None:
                active[0].append(entry)
