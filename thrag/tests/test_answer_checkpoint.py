"""Checkpoint replacement must preserve existing answers if a write fails."""

import json
from pathlib import Path

import pytest

from generate.batch_utils import save_answer_checkpoint


def test_checkpoint_keeps_pending_slots_in_original_order(tmp_path) -> None:
    path = tmp_path / "answers_temp.json"
    results = [None, {"query": "Question", "result": "Answer"}, None]

    save_answer_checkpoint(path, results)

    assert json.loads(path.read_text(encoding="utf-8")) == results


def test_failed_replacement_preserves_last_valid_checkpoint(tmp_path, monkeypatch) -> None:
    path = tmp_path / "answers_temp.json"
    original = '[{"query": "Question", "result": "Saved answer"}, null]'
    path.write_text(original, encoding="utf-8")

    def fail_replace(self, target):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        save_answer_checkpoint(path, [{"query": "Question", "result": "New answer"}, None])

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]
