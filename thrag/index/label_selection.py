"""Bound remote label selection to a short JSON array of candidate IDs."""

from __future__ import annotations

import json
import time

from openai import APIStatusError, OpenAI

from config import get_config
from model_services import create_answer_completion

# Selection should not occupy a question worker for ten oversized generations.
MAX_SELECTION_ATTEMPTS = 3


def format_numbered_candidates(labels: list[str]) -> str:
    return json.dumps(
        [{"id": index, "label": label} for index, label in enumerate(labels, start=1)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_selected_ids(
    content: str,
    *,
    key: str,
    candidate_count: int,
    max_choices: int,
) -> list[int]:
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {key}:
        raise ValueError(f"Expected a JSON object containing only {key!r}.")
    chosen = payload[key]
    if not isinstance(chosen, list) or not 1 <= len(chosen) <= max_choices:
        raise ValueError(f"{key!r} must contain between 1 and {max_choices} candidate IDs.")
    if any(type(value) is not int or not 1 <= value <= candidate_count for value in chosen):
        raise ValueError(f"{key!r} contains an invalid candidate ID.")

    # Keep the original graph order, as the label-based selectors did.
    selected = set(chosen)
    return [index for index in range(1, candidate_count + 1) if index in selected]


def select_labels(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    key: str,
    labels: list[str],
    max_choices: int,
    max_retries: int,
    description: str,
) -> list[str]:
    if not labels:
        return []
    config = get_config()
    max_choices = max(1, min(max_choices, len(labels)))
    attempts = max(1, min(max_retries, MAX_SELECTION_ATTEMPTS))
    # Only numbers are returned; output length no longer depends on label length.
    output_budget = max(128, 32 + max_choices * (len(str(len(labels))) + 8))
    schema = {
        "type": "object",
        "properties": {
            key: {
                "type": "array",
                "items": {"type": "integer", "enum": list(range(1, len(labels) + 1))},
                "minItems": 1,
                "maxItems": max_choices,
            }
        },
        "required": [key],
        "additionalProperties": False,
    }
    # Own the retry budget here, rather than multiplying it by SDK HTTP retries.
    selection_client = client.with_options(max_retries=0)

    for attempt in range(1, attempts + 1):
        finish_reason = None
        content = ""
        completion_tokens = None
        request_prompt = prompt
        if attempt > 1:
            request_prompt += (
                f"\nReturn only a compact JSON object with {key!r} containing "
                f"at most {max_choices} integer candidate IDs. Do not repeat labels or explain."
            )
        try:
            response = create_answer_completion(
                selection_client,
                model=model,
                messages=[
                    {"role": "system", "content": "Select relevant candidate IDs. Return only the required JSON."},
                    {"role": "user", "content": request_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": f"selected_{key}", "strict": True, "schema": schema},
                },
                temperature=config.answer_temperature,
                max_tokens=output_budget,
                extra_body=config.llm_extra_body(),
            )
            completion_tokens = getattr(getattr(response, "usage", None), "completion_tokens", None)
            if not response.choices:
                raise ValueError("Selection response contained no choices.")
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            content = choice.message.content or ""
            if finish_reason == "length":
                raise ValueError(f"Selection response was truncated at max_tokens={output_budget}.")
            if finish_reason not in {None, "stop"}:
                raise ValueError(f"Unexpected finish_reason={finish_reason!r}.")
            if not content.strip():
                raise ValueError("Selection response was empty.")
            if len(content) > max(4096, output_budget * 16):
                raise ValueError("Selection response exceeded the bounded ID format.")
            selected = parse_selected_ids(
                content, key=key, candidate_count=len(labels), max_choices=max_choices,
            )
            return [labels[index - 1] for index in selected]
        except Exception as exc:
            print(
                f"{description} attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {str(exc)[:500]} "
                f"(finish_reason={finish_reason!r}, response_chars={len(content)}, "
                f"completion_tokens={completion_tokens}, max_tokens={output_budget}, "
                f"candidates={len(labels)})",
                flush=True,
            )
            # Authentication, schema, model and context-window errors are not transient.
            if isinstance(exc, APIStatusError) and exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                raise
            if attempt < attempts:
                time.sleep(max(0.0, config.retry_backoff) * 2 ** (attempt - 1))

    print(f"{description} fell back to the first available {key} after {attempts} attempts.", flush=True)
    return labels[:max_choices]
