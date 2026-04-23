"""Tests for structured orchestration result and safety gating."""

from dataclasses import dataclass
import time

from biobank_agent.complexity import Strategy
from biobank_agent.llm import LLMResponse
from biobank_agent.orchestrator import (
    Claim,
    DebateProposal,
    EvidenceLink,
    ModelSpec,
    MultiModelOrchestrator,
    OrchestrationResult,
)


@dataclass
class _Rec:
    key_results: dict
    skill: str = "train_model"
    args: dict = None
    timestamp: str = "2026-04-23T12:00:00"


def test_orchestration_result_llm_compat_properties():
    raw = LLMResponse(text="hello world", tool_calls=[], usage={"prompt_tokens": 1})
    out = OrchestrationResult(final_answer="hello world", llm_response=raw)
    assert out.text == "hello world"
    assert out.has_tool_calls is False
    assert out.usage["prompt_tokens"] == 1


def test_safety_gate_marks_partial_without_evidence():
    orch = MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="",
        default_model="test-model",
    )
    raw = LLMResponse(text="We prove causal effect.", tool_calls=[], usage={})
    out = OrchestrationResult(
        final_answer=raw.text,
        claims=[Claim(claim_id="c1", text="Causal claim", confidence=0.8)],
        evidence_links=[],
        debate_trace={"disagreement": True},
        safety_status="PASS",
        llm_response=raw,
    )
    orch._apply_execution_judge(out, records=[_Rec(key_results={"n_cases": 50})])
    assert out.safety_status == "PARTIAL"
    assert "[Adjudication Required]" in out.text


def test_safety_gate_does_not_block_low_risk_without_records():
    orch = MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="",
        default_model="test-model",
    )
    raw = LLMResponse(text="Here is a draft idea for next steps.", tool_calls=[], usage={})
    out = OrchestrationResult(
        final_answer=raw.text,
        claims=[Claim(claim_id="c1", text="draft idea", confidence=0.4)],
        evidence_links=[],
        debate_trace={"strategy": "single", "disagreement": False},
        safety_status="PASS",
        llm_response=raw,
    )
    orch._apply_execution_judge(out, records=[])
    assert out.safety_status == "PASS"


def test_parallel_chat_timeout_returns_without_hanging(monkeypatch):
    orch = MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="",
        default_model="m1",
        model_pool=[ModelSpec("m1"), ModelSpec("m2")],
        parallel_timeout_s=0.1,
    )

    class SlowClient:
        def chat(self, messages, tools=None):
            time.sleep(1.0)
            return LLMResponse(text="slow", tool_calls=[], usage={})

    monkeypatch.setattr(orch, "get_client", lambda model_id: SlowClient())

    t0 = time.time()
    proposals = orch._parallel_chat(orch.model_pool, [{"role": "user", "content": "x"}], None)
    elapsed = time.time() - t0
    assert elapsed < 0.8
    assert proposals == []


def test_safety_gate_passes_when_execution_evidence_exists():
    orch = MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="",
        default_model="test-model",
    )
    raw = LLMResponse(text="We observed hazard ratio 1.2 and p < 0.01.", tool_calls=[], usage={})
    out = OrchestrationResult(
        final_answer=raw.text,
        claims=[Claim(claim_id="c1", text="hazard ratio indicates association", confidence=0.8)],
        evidence_links=[
            EvidenceLink(
                claim_id="c1",
                evidence_type="execution_record",
                evidence_id="2026-01-01:statistical_review",
                relation="supports",
                score=0.8,
            ),
        ],
        debate_trace={"strategy": "debate", "disagreement": True},
        safety_status="PASS",
        llm_response=raw,
    )
    orch._apply_execution_judge(out, records=[_Rec(key_results={"n_cases": 500})])
    assert out.safety_status == "PASS"
    assert "[Adjudication Required]" not in out.text


def test_route_rolls_back_to_ensemble_on_gate_fail(monkeypatch):
    orch = MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="",
        default_model="m1",
        model_pool=[ModelSpec("m1"), ModelSpec("m2")],
    )
    calls: list[Strategy] = []

    def fake_run(strategy, query, messages, tools, records):
        calls.append(strategy)
        if strategy == Strategy.DEBATE:
            raw = LLMResponse(text="We prove causality.", tool_calls=[], usage={})
            return OrchestrationResult(
                final_answer=raw.text,
                claims=[Claim(claim_id="c1", text="causal claim", confidence=0.9)],
                evidence_links=[],
                debate_trace={"strategy": "debate"},
                safety_status="PARTIAL",
                llm_response=raw,
            )
        raw = LLMResponse(text="Fallback response.", tool_calls=[], usage={})
        return OrchestrationResult(
            final_answer=raw.text,
            claims=[],
            evidence_links=[],
            debate_trace={"strategy": strategy.value},
            safety_status="PASS",
            llm_response=raw,
        )

    monkeypatch.setattr(orch, "_run_strategy", fake_run)
    monkeypatch.setattr(orch, "_attach_execution_evidence", lambda out, records: None)
    monkeypatch.setattr(orch, "_apply_execution_judge", lambda out, records: None)

    out = orch.route(
        query="complex question",
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        records=[],
        force_strategy=Strategy.DEBATE,
    )

    assert calls == [Strategy.DEBATE, Strategy.ENSEMBLE]
    assert out.debate_trace.get("rollback_from") == "debate"
    assert out.debate_trace.get("rollback_chain") == ["debate", "ensemble"]


def test_debate_forces_adjudication_tools_when_disagreement_and_no_execution(monkeypatch):
    orch = MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="",
        default_model="judge-model",
        model_pool=[
            ModelSpec("m1", role="generalist"),
            ModelSpec("m2", role="generalist"),
            ModelSpec("judge-model", role="judge"),
        ],
    )
    proposals = [
        DebateProposal(model_id="m1", response=LLMResponse(text="Hypothesis A", tool_calls=[], usage={})),
        DebateProposal(model_id="m2", response=LLMResponse(text="Hypothesis B", tool_calls=[], usage={})),
    ]
    monkeypatch.setattr(orch, "_parallel_chat", lambda models, messages, tools: proposals)
    monkeypatch.setattr(
        orch,
        "_anonymous_initial_votes",
        lambda voters, proposals: [
            {"voter_model": "m1", "vote": "A", "confidence": 0.7, "reason": "x"},
            {"voter_model": "m2", "vote": "B", "confidence": 0.8, "reason": "y"},
        ],
    )
    monkeypatch.setattr(
        orch,
        "_suggest_reretrieval_queries",
        lambda query, proposals: ["cardiovascular risk confounder adjustment"],
    )
    monkeypatch.setattr(
        orch,
        "_judge_select",
        lambda judge_spec, proposals, messages, tools: LLMResponse(
            text="Final answer without tools.",
            tool_calls=[],
            usage={},
        ),
    )

    out = orch.debate(
        query="Comprehensive cardiovascular risk analysis",
        messages=[{"role": "user", "content": "x"}],
        tools=[
            {"type": "function", "function": {"name": "statistical_review", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "safety_check", "parameters": {"type": "object"}}},
        ],
        records=[],
    )

    assert out.has_tool_calls is True
    assert any(tc.id.startswith("adjudicate_") for tc in out.tool_calls)
    assert [tc.name for tc in out.tool_calls] == ["statistical_review", "safety_check"]
    assert out.debate_trace.get("forced_adjudication") is True
    assert out.safety_status == "PARTIAL"
