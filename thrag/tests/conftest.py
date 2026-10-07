"""Pytest configuration for TH-RAG."""

import os
from tempfile import TemporaryDirectory


def pytest_configure(config) -> None:
    """Keep tests away from the server's research data and results."""
    storage = TemporaryDirectory(prefix="thrag-pytest-")
    previous_root = os.environ.get("THRAG_STORAGE_ROOT")
    os.environ["THRAG_STORAGE_ROOT"] = storage.name

    def cleanup() -> None:
        if previous_root is None:
            os.environ.pop("THRAG_STORAGE_ROOT", None)
        else:
            os.environ["THRAG_STORAGE_ROOT"] = previous_root
        storage.cleanup()

    config.add_cleanup(cleanup)
