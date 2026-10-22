"""Prepare the UltraDomain Agriculture subset from its released JSONL.

Reads local data only. Writes exactly 100 QA records; this source has exactly
100, so file order is preserved. Passages are deduplicated by exact context
text. No model calls or downloads.
"""

from __future__ import annotations

from _prepare_ultradomain import run_domain


if __name__ == "__main__":
    raise SystemExit(run_domain("agriculture"))
