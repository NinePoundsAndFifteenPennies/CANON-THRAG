"""Short-answer GraphRAG wrapper for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import get_config
from generate.graph_rag import GraphRAG as BaseGraphRAG
from generate.short_prompt import ShortAnswerPrompt, load_short_prompt
from prompt.answer_short import ANSWER_PROMPT, SHORT_SYSTEM_PROMPT


class GraphRAG(BaseGraphRAG):
    def __init__(
        self, dataset_name: str, *, short_prompt: ShortAnswerPrompt | str | Path | None = None,
    ) -> None:
        self.short_prompt = load_short_prompt(short_prompt)
        config = get_config(dataset_name)
        super().__init__(
            dataset_name=dataset_name,
            answer_prompt=self.short_prompt.answer_prompt,
            system_prompt=self.short_prompt.system_prompt,
            default_top_k1=config.top_k1,
            default_top_k2=config.top_k2,
            temperature=0.0,
            max_output_tokens=max(64, min(256, config.answer_max_tokens)),
        )

