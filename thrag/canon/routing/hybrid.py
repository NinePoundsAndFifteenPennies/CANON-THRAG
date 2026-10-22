"""Explicit composition: A proposes candidates and B selects their paths."""

from __future__ import annotations

from typing import Any

from canon.core.config import CanonConfig
from canon.core.contracts import QueryContext, RoutingResult
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts
from canon.routing.diffusion import DiffusionRouter
from canon.routing.preview import PreviewSelector


class HybridRouter:
    def __init__(self, store: EvidenceStore, hierarchy: HierarchyArtifacts, config: CanonConfig, runtime: Any, client: Any) -> None:
        self.diffusion = DiffusionRouter(store, hierarchy, config, runtime)
        self.selector = PreviewSelector(store, hierarchy, config, runtime, client)

    def route(self, context: QueryContext) -> RoutingResult:
        scores, diagnostics = self.diffusion.score(context)
        return self.selector.select(context, scores, "hybrid", diagnostics)
