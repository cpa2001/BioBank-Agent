"""Smart context compaction.

Replaces the legacy ``[:8000]`` truncation in ``agent.py:517`` with a
three-strategy pipeline:

1. ``structured_extract`` — for tool results that are dicts: keep
   numeric scalars and ``key_results`` style fields, drop large text
   blobs and pandas representations. Cheap, deterministic.

2. ``summarize_with_llm`` — for prose results above a threshold,
   ask the active LLM to compact while keeping numbers.

3. ``before_last_user_message`` — context-window
   compaction. Once the running token estimate exceeds ``high_water``
   percent of ``context_window``, summarize the conversation slice
   *before* the most recent user turn so the model still sees the
   active question with full fidelity.

Trigger thresholds (defaults, tunable in Settings):
- 60%  : warn (emit ``CONTEXT_WINDOW_WARNING`` event)
- 75%  : trigger smart compaction
- 90%  : forced aggressive compaction (drop everything except last
         user turn + most recent two tool exchanges)

Token counting uses a lightweight char/4 heuristic so we never
hard-depend on tiktoken; if tiktoken is importable we use it for
better fidelity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# ── Token estimation ─────────────────────────────────────────


def _import_tiktoken():
    try:
        import tiktoken  # type: ignore

        return tiktoken
    except Exception:  # pragma: no cover
        return None


_tiktoken = _import_tiktoken()
_encoder = None
if _tiktoken is not None:
    try:
        _encoder = _tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover
        _encoder = None


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for ``text``."""
    if not text:
        return 0
    if _encoder is not None:
        try:
            return len(_encoder.encode(text))
        except Exception:
            pass
    # 4 chars/token is OpenAI's documented heuristic for English-heavy
    # content. Biobank tool outputs are mostly numeric tables which
    # tokenize sparser, but undercounting is the safe direction here.
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    """Estimate cumulative tokens for a chat-message list."""
    total = 0
    for m in messages:
        role = str(m.get("role", "")) or "user"
        total += estimate_tokens(role) + 4  # role + JSON framing overhead
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(json.dumps(part, default=str))
        tool_calls = m.get("tool_calls") or []
        for tc in tool_calls:
            total += estimate_tokens(json.dumps(tc, default=str))
    return total


# ── Tool-result compaction ──────────────────────────────────


_LARGE_RESULT_KEYS = ("raw", "raw_text", "html", "markdown", "trace", "stdout", "stderr")
_PRIORITY_KEYS = (
    "n_cases", "n_controls", "n_total", "n_subjects", "n",
    "auc", "auc_ci", "mean_auc", "p_value", "log_rank_p",
    "or", "hr", "rr", "ci_low", "ci_high", "effect_size",
    "top_features", "top_associations",
    "summary", "warnings", "requires_repair", "error",
    "key_results", "scientific_finding", "disclosed",
)


@dataclass
class CompactedResult:
    text: str
    strategy: str
    original_tokens: int
    compacted_tokens: int
    dropped_keys: list[str]


def _stringify(value: Any, max_chars: int) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str, ensure_ascii=False)[:max_chars]
        except Exception:
            return str(value)[:max_chars]
    return str(value)[:max_chars]


def structured_extract(
    result: Any,
    *,
    target_tokens: int = 1500,
) -> CompactedResult:
    """Compact a structured tool result while keeping scientific signal.

    Strategy:
        1. If ``result`` is a dict, drop known large-payload keys
           (raw / html / markdown / etc.) up front.
        2. Keep priority keys verbatim.
        3. Anything left gets stringified with a per-value char budget.
        4. If we still exceed ``target_tokens`` after that, keep only
           the priority keys.
    """
    if not isinstance(result, dict):
        text = _stringify(result, max_chars=target_tokens * 4)
        return CompactedResult(
            text=text,
            strategy="stringify",
            original_tokens=estimate_tokens(_stringify(result, max_chars=10**9)),
            compacted_tokens=estimate_tokens(text),
            dropped_keys=[],
        )

    full_text = _stringify(result, max_chars=10**9)
    original_tokens = estimate_tokens(full_text)

    dropped: list[str] = []
    pruned: dict[str, Any] = {}
    for key, value in result.items():
        if key in _LARGE_RESULT_KEYS and isinstance(value, str) and len(value) > 800:
            dropped.append(key)
            continue
        pruned[key] = value

    text = _stringify(pruned, max_chars=10**9)
    if estimate_tokens(text) > target_tokens:
        # Aggressive fallback: keep only priority keys.
        priority_only = {k: pruned[k] for k in _PRIORITY_KEYS if k in pruned}
        # Always keep at least a top-level shape so the model can decide
        # whether to ask for more.
        priority_only.setdefault("_shape", list(pruned.keys()))
        text = _stringify(priority_only, max_chars=target_tokens * 4)

    return CompactedResult(
        text=text,
        strategy="structured_extract",
        original_tokens=original_tokens,
        compacted_tokens=estimate_tokens(text),
        dropped_keys=dropped,
    )


# ── Conversation-level compaction ───────────────────────────


@dataclass
class CompactionDecision:
    """Outcome of evaluating whether the conversation needs compaction."""

    should_compact: bool
    severity: str  # "noop" | "warn" | "compact" | "force"
    estimated_tokens: int
    high_water_tokens: int
    context_window: int
    reason: str = ""


def evaluate(
    messages: list[dict],
    *,
    context_window: int,
    warn_pct: float = 0.6,
    compact_pct: float = 0.75,
    force_pct: float = 0.9,
) -> CompactionDecision:
    """Decide whether ``messages`` warrant compaction."""

    estimated = estimate_messages_tokens(messages)
    if context_window <= 0:
        return CompactionDecision(False, "noop", estimated, 0, 0)
    pct = estimated / context_window
    if pct >= force_pct:
        severity = "force"
        should = True
        reason = f"context at {pct:.0%} (force threshold {force_pct:.0%})"
    elif pct >= compact_pct:
        severity = "compact"
        should = True
        reason = f"context at {pct:.0%} (compact threshold {compact_pct:.0%})"
    elif pct >= warn_pct:
        severity = "warn"
        should = False
        reason = f"context at {pct:.0%} (warn threshold {warn_pct:.0%})"
    else:
        severity = "noop"
        should = False
        reason = ""
    return CompactionDecision(
        should_compact=should,
        severity=severity,
        estimated_tokens=estimated,
        high_water_tokens=int(context_window * compact_pct),
        context_window=context_window,
        reason=reason,
    )


SummarizeFn = Callable[[list[dict]], str]


def before_last_user_message(
    messages: list[dict],
    *,
    summarize: Optional[SummarizeFn] = None,
    target_tokens: int = 800,
) -> list[dict]:
    """Replace pre-last-user-turn content with a summary.

    The most recent ``user`` turn (and everything after it) is preserved
    verbatim so the model never sees a degraded "current question".

    If ``summarize`` is provided it's called with the slice to summarise;
    otherwise we fall back to a deterministic heuristic that keeps the
    first system message and a structured extract of every tool/assistant
    turn before the last user turn.
    """
    last_user_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i
    if last_user_idx <= 0:
        return list(messages)

    head = messages[:last_user_idx]
    tail = messages[last_user_idx:]
    if not head:
        return list(messages)
    # If head is only system messages, there's nothing meaningful to compact.
    if all(m.get("role") == "system" for m in head):
        return list(messages)

    if summarize is not None:
        try:
            summary_text = summarize(head)
        except Exception as e:
            logger.warning("compaction summarize callback failed: %s", e)
            summary_text = _heuristic_summary(head)
    else:
        summary_text = _heuristic_summary(head)

    # Cap the summary itself.
    while estimate_tokens(summary_text) > target_tokens and len(summary_text) > 200:
        summary_text = summary_text[: int(len(summary_text) * 0.8)]

    system_msgs = [m for m in head if m.get("role") == "system"]
    summary_msg = {
        "role": "system",
        "content": (
            "## Compacted earlier conversation\n"
            "The earlier turns have been summarised to free context. "
            "Treat the summary as authoritative for facts; consult tool "
            "results in subsequent turns for exact numbers.\n\n"
            f"{summary_text}"
        ),
    }
    return system_msgs + [summary_msg] + tail


def force_aggressive(messages: list[dict], *, keep_last_n_tool_pairs: int = 2) -> list[dict]:
    """Last-resort compaction — preserves the active tool exchange.

    Keep:
        - All system messages.
        - Everything from the most recent user turn onward (active tool
          requests + their tool results that the LLM still needs to see
          in the next round).
        - The last ``keep_last_n_tool_pairs`` (assistant tool_call,
          tool result) pairs immediately preceding the last user turn.

    The entire tail (>= the last user turn) is kept verbatim, including any
    assistant tool_calls + tool results already produced in the current round.
    Cutting at the most recent user turn alone could silently drop that fresh
    active context.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    # Find last user message.
    last_user_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i
    if last_user_idx < 0:
        return system_msgs + list(messages[-2:])

    # Tail = last user turn AND any assistant/tool messages produced
    # since (the runtime is mid-round when force compaction triggers).
    tail = list(messages[last_user_idx:])

    # Walk backwards from last_user_idx-1 collecting pairs.
    pairs: list[dict] = []
    pair_count = 0
    i = last_user_idx - 1
    while i >= 0 and pair_count < keep_last_n_tool_pairs:
        m = messages[i]
        if m.get("role") == "tool":
            pairs.insert(0, m)
            # Walk back to its assistant tool_call message.
            j = i - 1
            while j >= 0 and not (
                messages[j].get("role") == "assistant"
                and messages[j].get("tool_calls")
            ):
                j -= 1
            if j >= 0:
                pairs.insert(0, messages[j])
                pair_count += 1
                i = j - 1
                continue
        i -= 1
    return system_msgs + pairs + tail


# ── Heuristic summary for offline / no-LLM mode ─────────────


def _heuristic_summary(messages: list[dict]) -> str:
    """Deterministic summary used when no LLM summarizer is available."""
    bullets: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            continue
        if role == "user" and isinstance(content, str):
            bullets.append(f"- user: {content[:160]}")
        elif role == "assistant" and isinstance(content, str) and content:
            bullets.append(f"- assistant: {content[:160]}")
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(
                tc.get("function", {}).get("name", "")
                for tc in m.get("tool_calls", [])
            )
            bullets.append(f"- assistant requested tools: {names}")
        elif role == "tool":
            text = content if isinstance(content, str) else _stringify(content, max_chars=400)
            bullets.append(f"- tool result: {text[:240]}")
    if not bullets:
        return "(empty earlier turn)"
    return "\n".join(bullets[-30:])


__all__ = [
    "CompactedResult",
    "CompactionDecision",
    "structured_extract",
    "evaluate",
    "before_last_user_message",
    "force_aggressive",
    "estimate_tokens",
    "estimate_messages_tokens",
]
