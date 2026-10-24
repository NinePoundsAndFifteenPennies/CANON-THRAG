"""Write optional module-switch task lists; do not execute or score experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

from canon.core.config import CanonConfig
from canon.core.io import atomic_json
from generate.short_prompt import add_short_prompt_argument, load_short_prompt


def variants(matrix: str, base: CanonConfig) -> list[tuple[str, CanonConfig]]:
    if matrix == "core":
        return [(f"h{h}_r{r}_e{e}", replace(
            base, hierarchy=base.hierarchy if h else "none", soft_links=base.soft_links if h else False,
            routing=base.routing if r else "baseline", bridge=base.bridge if e else False, balance=base.balance if e else False,
        )) for h, r, e in product((0, 1), repeat=3)]
    if matrix == "routing":
        return [(mode, replace(base, routing=mode)) for mode in ("baseline", "a", "b", "hybrid")]
    if matrix == "hierarchy":
        return [(name, replace(base, hierarchy=mode, soft_links=soft)) for name, mode, soft in (
            ("none", "none", False), ("split", "split", False), ("split_soft", "split", True),
            ("split_merge", "split_merge", False), ("split_merge_soft", "split_merge", True),
        )]
    if matrix == "allocation":
        return [(f"bridge{b}_balance{c}", replace(base, bridge=bool(b), balance=bool(c))) for b, c in product((0, 1), repeat=2)]
    raise ValueError(f"Unknown matrix: {matrix}")


def main() -> None:
    from config import get_config
    from canon.pipeline import safe_name

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=safe_name, required=True)
    parser.add_argument("--matrix", choices=("core", "routing", "hierarchy", "allocation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_short_prompt_argument(parser)
    args = parser.parse_args()
    get_config(args.dataset)  # Load the same .env as the server pipeline.
    short_prompt = load_short_prompt(args.short_prompt)
    tasks = [{
        "name": name,
        "argv": ["python", "-m", "canon.pipeline", "--dataset", args.dataset, "--short-prompt", short_prompt.source],
        "environment": {f"CANON_{key.upper()}": str(value).lower() if type(value) is bool else str(value) for key, value in method.to_dict().items()},
    } for name, method in variants(args.matrix, CanonConfig.from_env())]
    atomic_json(args.output, {
        "dataset": args.dataset, "matrix": args.matrix, "tasks": tasks,
        "short_prompt": short_prompt.source, "short_prompt_identity": short_prompt.metadata(),
    })
    print(f"Wrote {len(tasks)} task definitions to {args.output}; no jobs were executed")


if __name__ == "__main__":
    main()
