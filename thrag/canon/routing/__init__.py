"""Select a routing strategy without coupling A to B."""

from typing import Any

from canon.core.config import CanonConfig
from canon.core.contracts import Router
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts


def make_router(store: EvidenceStore, hierarchy: HierarchyArtifacts, config: CanonConfig, runtime: Any, client: Any = None) -> Router:
    if config.routing == "a":
        from canon.routing.diffusion import DiffusionRouter
        return DiffusionRouter(store, hierarchy, config, runtime)
    if client is None:
        raise ValueError(f"Routing mode {config.routing!r} requires a chat client")
    if config.routing == "baseline":
        from canon.routing.baseline import BaselineRouter
        return BaselineRouter(hierarchy, runtime, client)
    if config.routing == "b":
        from canon.routing.preview import EvidencePreviewRouter
        return EvidencePreviewRouter(store, hierarchy, config, runtime, client)
    from canon.routing.hybrid import HybridRouter
    return HybridRouter(store, hierarchy, config, runtime, client)
