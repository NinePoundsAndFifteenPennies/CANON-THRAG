"""Load and identify a short-answer template selected through the CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHORT_PROMPT = PROJECT_ROOT / "prompt" / "answer_short.py"


@dataclass(frozen=True)
class ShortAnswerPrompt:
    source: str
    file_sha256: str
    answer_prompt: str
    system_prompt: str
    sha256: str

    def metadata(self) -> dict[str, str]:
        return {"source": self.source, "file_sha256": self.file_sha256, "sha256": self.sha256}

    def texts(self) -> dict[str, str]:
        return {"answer_prompt": self.answer_prompt, "system_prompt": self.system_prompt}

    def snapshot(self) -> dict[str, str]:
        return {**self.metadata(), **self.texts()}

    def matches(self, metadata: Any) -> bool:
        return isinstance(metadata, dict) and (
            metadata.get("source") == self.source and metadata.get("sha256") == self.sha256
        )


def add_short_prompt_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--short-prompt", type=Path, metavar="FILE.py",
        help="Short-answer template file; relative to thrag/. Default: prompt/answer_short.py.",
    )


def _read_template(path: Path) -> tuple[dict[str, Any], str]:
    if path.suffix.lower() != ".py" or not path.is_file():
        raise ValueError(f"Short-answer template must be an existing .py file: {path}")
    source = path.read_bytes()
    values = runpy.run_path(str(path))
    if path.read_bytes() != source:
        raise ValueError(f"Short-answer template changed while loading: {path}")
    return values, hashlib.sha256(source).hexdigest()


def _require_text(values: dict[str, Any], name: str, path: Path) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must define a nonempty string named {name}.")
    return value


def load_short_prompt(
    selection: ShortAnswerPrompt | str | Path | None = None,
) -> ShortAnswerPrompt:
    if isinstance(selection, ShortAnswerPrompt):
        return selection
    path = DEFAULT_SHORT_PROMPT if selection is None else Path(selection).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    values, file_sha256 = _read_template(path)
    answer_prompt = _require_text(values, "ANSWER_PROMPT", path)
    for placeholder in ("{{context}}", "{{question}}"):
        if placeholder not in answer_prompt:
            raise ValueError(f"{path}: ANSWER_PROMPT must contain {placeholder}.")
    if "SHORT_SYSTEM_PROMPT" in values:
        system_prompt = _require_text(values, "SHORT_SYSTEM_PROMPT", path)
    else:
        defaults, _ = _read_template(DEFAULT_SHORT_PROMPT)
        system_prompt = _require_text(defaults, "SHORT_SYSTEM_PROMPT", DEFAULT_SHORT_PROMPT)
    source = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
    encoded = json.dumps(
        {"answer_prompt": answer_prompt, "system_prompt": system_prompt},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return ShortAnswerPrompt(source, file_sha256, answer_prompt, system_prompt, hashlib.sha256(encoded).hexdigest())


def check_saved_short_prompt(
    answer_file: Path, selected: ShortAnswerPrompt, *, allow_untracked: bool = False,
) -> None:
    """Check the identity of cached answers before a stage is skipped."""
    with answer_file.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    message = f"Short-answer prompt identity is not verified for {answer_file}; use --force to generate the selected template."
    if not isinstance(rows, list):
        raise ValueError(message)
    tracked = untracked = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(message)
        meta = row.get("meta", {})
        recorded = meta.get("short_prompt") if isinstance(meta, dict) else None
        if recorded is None:
            untracked += 1
        elif selected.matches(recorded):
            tracked += 1
        else:
            raise ValueError(message)
    if (untracked and (tracked or not allow_untracked)) or (not rows and not allow_untracked):
        raise ValueError(message)
