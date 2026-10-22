"""Prepare the UltraDomain Legal subset from its released JSONL.

Reads local data only. Selects 100 QA records without replacement with seed 42;
all released passages stay in the corpus. Passages are deduplicated by exact
context text. Duplicate source questions are retained and flagged in metadata.
No model calls or downloads.
"""

from __future__ import annotations

from _prepare_ultradomain import run_domain


if __name__ == "__main__":
    raise SystemExit(run_domain("legal"))
