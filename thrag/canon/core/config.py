"""Independent method switches. Deployment settings remain in config.py."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import os
from typing import Any


def parse_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "on"}:
        return True
    if value.lower() in {"false", "0", "off"}:
        return False
    raise ValueError(f"Expected on/off or true/false, got {value!r}")


@dataclass(frozen=True)
class CanonConfig:
    sentence_view: str = "raw"
    routing: str = "hybrid"
    bridge: bool = True
    balance: bool = True
    seed_sentences: int = 32
    seed_labels: int = 32
    bottom_up_weight: float = 0.50
    restart: float = 0.15
    diffusion_iterations: int = 100
    diffusion_tolerance: float = 1e-6
    activation_label_weight: float = 0.50
    activation_top_n: int = 3
    candidate_topics: int = 0
    candidate_subtopics: int = 0
    preview_input_tokens: int = 10240
    preview_sentence_tokens: int = 32
    preview_entities: int = 2
    route_attempts: int = 3
    bridge_seed_sentences: int = 5
    bridge_weight: float = 0.20
    coverage_weight: float = 0.30
    coverage_anchor_fraction: float = 0.50

    def __post_init__(self) -> None:
        for name, allowed in {
            "sentence_view": {"raw", "triple", "llm"},
            "routing": {"baseline", "a", "b", "hybrid"},
        }.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"CANON_{name.upper()} must be one of {sorted(allowed)}")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"CANON_{field.name.upper()} must be finite")
            if type(value) is int:
                minimum = 0 if field.name.startswith("candidate_") else 1
                if value < minimum:
                    raise ValueError(f"CANON_{field.name.upper()} must be >= {minimum}")
        for name in (
            "bottom_up_weight", "activation_label_weight", "bridge_weight", "coverage_weight",
            "coverage_anchor_fraction",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"CANON_{name.upper()} must be between 0 and 1")
        if not 0 < self.restart < 1 or self.diffusion_tolerance <= 0:
            raise ValueError("CANON_RESTART must be in (0,1); diffusion tolerance must be positive")
        if self.route_attempts > 3:
            raise ValueError("CANON_ROUTE_ATTEMPTS cannot exceed 3")

    @classmethod
    def from_env(cls, **overrides: Any) -> CanonConfig:
        values: dict[str, Any] = {}
        for field in fields(cls):
            raw = os.getenv(f"CANON_{field.name.upper()}")
            if raw is not None:
                parser = parse_bool if type(field.default) is bool else type(field.default)
                values[field.name] = parser(raw.strip())
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def routing_settings(self) -> dict[str, Any]:
        """Settings that must match when replaying a route for an E ablation."""
        return {key: value for key, value in asdict(self).items() if (
            key in {"routing", "sentence_view"}
            or key.startswith(("seed_", "diffusion_", "activation_", "candidate_", "preview_", "route_"))
            or key in {"bottom_up_weight", "restart"}
        )}
