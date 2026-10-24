"""Progress and atomic checkpoints shared by short and long answer batches."""

from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

from tqdm import tqdm


def completed_answers(futures: list[Future], description: str) -> Iterator[Future]:
    pending = set(futures)
    total = len(pending)
    completed = 0
    with tqdm(total=total, desc=description) as progress:
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                tqdm.write(
                    f"{description}: completed={completed}/{total}, pending={len(pending)}; "
                    "waiting for running/queued questions (pending slots remain null)."
                )
            for future in done:
                yield future
                completed += 1
                progress.update(1)


def save_answer_checkpoint(path: Path, results: list[dict[str, Any] | None]) -> None:
    save_json_atomic(path, results)


def save_json_atomic(path: Path, payload: Any) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
