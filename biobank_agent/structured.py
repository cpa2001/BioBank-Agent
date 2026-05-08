"""Structured output enforcement via instructor (optional).

Falls back to raw Pydantic validation + manual JSON parsing if instructor
is not installed. This keeps instructor truly optional while providing
the same API surface.

Source: github.com/jxnl/instructor
Reference: Claude research doc §C1: "Instructor (Pydantic-based) —
           production de-facto for typed outputs"

Usage:
    from biobank_agent.structured import extract_structured
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        answer: str
        confidence: float

    result = extract_structured(llm, MyOutput, "What is the prevalence of E11?")
    # result is a validated MyOutput instance
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


# ── Availability guard (same pattern as verification.py) ─────────────────────

try:
    import instructor
    _INSTRUCTOR_AVAILABLE = True
    logger.debug("instructor available — using patched extraction")
except ImportError:
    _INSTRUCTOR_AVAILABLE = False
    logger.debug("instructor not installed; using fallback Pydantic validation")


def is_instructor_available() -> bool:
    """Check if instructor is installed."""
    return _INSTRUCTOR_AVAILABLE


def extract_structured(
    llm: "LLMClient",
    response_model: Type[T],
    prompt: str,
    max_retries: int = 2,
    system_prompt: str = "You are a precise data extraction assistant. Output valid JSON only.",
    **kwargs: Any,
) -> T:
    """Extract a typed Pydantic object from LLM output.

    If instructor is available: uses instructor's patching + automatic retry
    on validation failure (instructor handles the retry logic).

    If instructor is NOT available: sends a structured prompt to the LLM,
    parses JSON from the response, and validates with Pydantic.

    Args:
        llm: The LLM client instance
        response_model: Pydantic model class to extract
        prompt: User prompt describing what to extract
        max_retries: Number of retries on validation failure (fallback mode)
        system_prompt: System prompt for the LLM
        **kwargs: Additional arguments passed to the LLM

    Returns:
        A validated instance of response_model

    Raises:
        ValueError: If extraction fails after all retries
        ValidationError: If the response doesn't match the schema (no more retries)
    """
    if _INSTRUCTOR_AVAILABLE:
        return _extract_with_instructor(llm, response_model, prompt, max_retries, system_prompt, **kwargs)
    else:
        return _extract_fallback(llm, response_model, prompt, max_retries, system_prompt, **kwargs)


# ── Instructor path ──────────────────────────────────────────────────────────


def _extract_with_instructor(
    llm: "LLMClient",
    response_model: Type[T],
    prompt: str,
    max_retries: int,
    system_prompt: str,
    **kwargs: Any,
) -> T:
    """Use instructor library for extraction with automatic retry + patching.

    Instructor wraps the OpenAI client to handle structured output extraction
    with automatic retry on validation errors.
    """
    try:
        # instructor patches the client to return Pydantic models
        client = instructor.from_openai(llm.client)
        result = client.chat.completions.create(
            model=kwargs.get("model", getattr(llm, "model", "gpt-4")),
            response_model=response_model,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return result
    except Exception as e:
        logger.warning("instructor extraction failed: %s, falling back to manual", e)
        return _extract_fallback(llm, response_model, prompt, max_retries, system_prompt, **kwargs)


# ── Fallback path (no instructor) ───────────────────────────────────────────


def _extract_fallback(
    llm: "LLMClient",
    response_model: Type[T],
    prompt: str,
    max_retries: int,
    system_prompt: str,
    **kwargs: Any,
) -> T:
    """Manual extraction: prompt LLM for JSON, parse, validate with Pydantic.

    Includes the Pydantic schema in the prompt to guide the LLM,
    then retries with error feedback on validation failure.
    """
    schema = response_model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    extraction_prompt = f"""{prompt}

Respond with a JSON object matching this schema:
{schema_str}

Output ONLY the JSON object, no explanation or markdown fences."""

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0 and last_error:
                # Include error feedback for retry
                extraction_prompt_with_error = (
                    f"{extraction_prompt}\n\n"
                    f"Previous attempt failed with: {last_error}\n"
                    f"Please fix the output to match the schema exactly."
                )
                current_prompt = extraction_prompt_with_error
            else:
                current_prompt = extraction_prompt

            response = llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_prompt},
                ],
                max_tokens=kwargs.get("max_tokens", 2000),
            )

            text = response.text.strip()

            # Strip markdown code fences if present
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    stripped = part.strip()
                    if stripped.startswith("json"):
                        stripped = stripped[4:].strip()
                    if stripped.startswith("{") or stripped.startswith("["):
                        text = stripped
                        break

            data = json.loads(text)
            return response_model.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.debug("Extraction attempt %d failed: %s", attempt + 1, e)
            continue

    raise ValueError(
        f"Failed to extract {response_model.__name__} after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
