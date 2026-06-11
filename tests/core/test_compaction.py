"""Unit tests for biobank_agent.core.compaction."""

from __future__ import annotations

from biobank_agent.core.compaction import (
    before_last_user_message,
    estimate_messages_tokens,
    estimate_tokens,
    evaluate,
    force_aggressive,
    structured_extract,
)


def test_estimate_tokens_nonzero():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0


def test_structured_extract_drops_large_raw_text():
    big = "x" * 5000
    result = {"raw": big, "auc": 0.91, "n_cases": 1234}
    out = structured_extract(result, target_tokens=200)
    assert "raw" in out.dropped_keys
    assert "0.91" in out.text
    assert "1234" in out.text
    assert out.compacted_tokens <= max(out.original_tokens, 200)


def test_structured_extract_keeps_priority_keys_under_pressure():
    result = {
        "auc": 0.83,
        "n_cases": 5000,
        "extra_blob_a": "z" * 4000,
        "extra_blob_b": "y" * 4000,
    }
    out = structured_extract(result, target_tokens=80)
    # Priority keys still present.
    assert "auc" in out.text
    assert "n_cases" in out.text


def test_evaluate_warn_compact_force():
    short = [{"role": "user", "content": "hi"}]
    decision = evaluate(short, context_window=4096)
    assert decision.severity == "noop"
    assert decision.should_compact is False

    big_msg = "x" * 9000
    long_msgs = [{"role": "user", "content": big_msg}] * 3
    decision_compact = evaluate(long_msgs, context_window=4096)
    assert decision_compact.severity in {"compact", "force"}
    assert decision_compact.should_compact is True


def test_before_last_user_message_keeps_tail_verbatim():
    messages = [
        {"role": "system", "content": "you are biobank agent"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    out = before_last_user_message(messages)
    # Last message is preserved verbatim
    assert out[-1] == {"role": "user", "content": "second question"}
    # System messages preserved
    assert out[0] == {"role": "system", "content": "you are biobank agent"}
    # A summary system message inserted
    assert any("Compacted earlier conversation" in (m.get("content") or "") for m in out)


def test_before_last_user_message_no_op_when_only_one_user_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "single"},
    ]
    out = before_last_user_message(messages)
    assert out == messages


def test_force_aggressive_keeps_recent_pairs_and_last_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t1"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "t1 result"},
        {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "t2"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "t2 result"},
        {"role": "user", "content": "q3"},
    ]
    out = force_aggressive(messages, keep_last_n_tool_pairs=2)
    # Has last user
    assert any(m == {"role": "user", "content": "q3"} for m in out)
    # Has system
    assert out[0]["role"] == "system"
    # Has both tool pairs
    tool_results = [m for m in out if m.get("role") == "tool"]
    assert len(tool_results) == 2


def test_force_aggressive_preserves_active_tool_round_after_user(monkeypatch=None):
    """Force compaction mid-round must preserve the active exchange.

    When force compaction triggers mid-round (i.e. after the runtime
    has already produced an assistant tool_call + tool result for the
    *current* user turn), the active exchange must be preserved.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "user", "content": "active q"},  # most recent user
        {"role": "assistant", "tool_calls": [{"id": "active_c1", "function": {"name": "prevalence"}}]},
        {"role": "tool", "tool_call_id": "active_c1", "content": "active result"},
    ]
    out = force_aggressive(messages, keep_last_n_tool_pairs=1)

    # Last user *and* its produced tool exchange must survive.
    roles = [m.get("role") for m in out]
    assert "user" in roles
    last_user_in_out = next(i for i, m in enumerate(out) if m.get("role") == "user")
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out[last_user_in_out:])
    assert any(
        m.get("role") == "tool" and m.get("tool_call_id") == "active_c1" for m in out[last_user_in_out:]
    )
