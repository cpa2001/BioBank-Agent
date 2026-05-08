"""Deterministic edge coverage for multi-model orchestration logic."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from biobank_agent.complexity import Strategy
from biobank_agent.llm import LLMResponse, ToolCall
from biobank_agent.orchestrator import (
    Claim,
    DebateProposal,
    EvidenceLink,
    ModelSpec,
    MultiModelOrchestrator,
    OrchestrationResult,
    SubagentCall,
    SubagentRole,
)


@dataclass
class _Rec:
    skill: str
    key_results: dict
    args: dict | None = None
    timestamp: str = "2026-05-08T00:00:00"


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append((messages, tools, kwargs))
        item = self.responses.pop(0) if self.responses else LLMResponse(text="fallback response.")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMResponse(text=item)
        return item


def _orch(model_pool=None):
    return MultiModelOrchestrator(
        base_url="http://example.com",
        api_key="key",
        default_model="m1",
        model_pool=model_pool or [ModelSpec("m1", priority=1), ModelSpec("m2", priority=2)],
        parallel_timeout_s=1.0,
    )


def test_orchestration_result_text_setter_and_to_dict():
    raw = LLMResponse(text="raw")
    out = OrchestrationResult(
        llm_response=raw,
        claims=[Claim("c1", "claim text")],
        evidence_links=[EvidenceLink("c1", "execution_record", "r1")],
    )

    out.text = "updated"
    data = out.to_dict()

    assert raw.text == "updated"
    assert out.tool_calls == []
    assert data["final_answer"] == "updated"
    assert data["claims"][0]["claim_id"] == "c1"


def test_get_client_caches_instances():
    orch = _orch()

    first = orch.get_client("m1")
    second = orch.get_client("m1")

    assert first is second


def test_route_uses_classifier_and_records_last_result(monkeypatch):
    orch = _orch([ModelSpec("m1")])
    fake_client = _Client([LLMResponse(text="A low risk planning answer with enough detail to form a claim.")])
    monkeypatch.setattr(orch, "get_client", lambda _model_id: fake_client)
    monkeypatch.setattr(
        "biobank_agent.orchestrator.classify_complexity",
        lambda query, records, threshold: SimpleNamespace(strategy=Strategy.SINGLE, score=0.1, reason="simple"),
    )

    out = orch.route("simple", [{"role": "user", "content": "q"}], records=None)

    assert out.debate_trace["strategy"] == "single"
    assert orch.last_result is out


def test_run_strategy_dispatches_and_falls_back(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2")])
    monkeypatch.setattr(orch, "debate", lambda **kwargs: OrchestrationResult(final_answer="debate"))
    monkeypatch.setattr(orch, "supervisor", lambda **kwargs: OrchestrationResult(final_answer="supervisor"))
    monkeypatch.setattr(orch, "ensemble", lambda **kwargs: OrchestrationResult(final_answer="ensemble"))
    fake_client = _Client([LLMResponse(text="single"), LLMResponse(text="fallback")])
    monkeypatch.setattr(orch, "get_client", lambda _model_id: fake_client)

    assert orch._run_strategy(Strategy.DEBATE, "q", [], None, []).text == "debate"
    assert orch._run_strategy(Strategy.SUPERVISOR, "q", [], None, []).text == "supervisor"
    assert orch._run_strategy(Strategy.ENSEMBLE, "q", [], None, []).text == "ensemble"
    assert orch._run_strategy(Strategy.SINGLE, "q", [], None, []).text == "single"
    assert orch._run_strategy(object(), "q", [], None, []).text == "fallback"


def test_rollback_no_candidates_and_failed_candidate(monkeypatch):
    single_orch = _orch([ModelSpec("m1")])
    failed = OrchestrationResult(final_answer="bad", safety_status="PARTIAL")
    assert single_orch._rollback_on_gate_fail(Strategy.SINGLE, "q", [], None, [], failed) is failed

    orch = _orch([ModelSpec("m1"), ModelSpec("m2")])
    calls = []

    def _run(strategy, query, messages, tools, records):
        calls.append(strategy)
        if strategy == Strategy.ENSEMBLE:
            raise RuntimeError("ensemble failed")
        return OrchestrationResult(final_answer="single ok", safety_status="PASS", debate_trace={"strategy": "single"})

    monkeypatch.setattr(orch, "_run_strategy", _run)
    monkeypatch.setattr(orch, "_attach_execution_evidence", lambda out, records: None)
    monkeypatch.setattr(orch, "_apply_execution_judge", lambda out, records: None)

    out = orch._rollback_on_gate_fail(Strategy.DEBATE, "q", [], None, [], failed)

    assert calls == [Strategy.ENSEMBLE, Strategy.SINGLE]
    assert out.text == "single ok"

    all_fail = _orch([ModelSpec("m1"), ModelSpec("m2")])
    monkeypatch.setattr(all_fail, "_run_strategy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no route")))
    assert all_fail._rollback_on_gate_fail(Strategy.DEBATE, "q", [], None, [], failed) is failed

    continue_orch = _orch([ModelSpec("m1"), ModelSpec("m2")])
    continued = []

    def _partial_then_pass(strategy, query, messages, tools, records):
        continued.append(strategy)
        if strategy == Strategy.ENSEMBLE:
            return OrchestrationResult(final_answer="partial", safety_status="PARTIAL", debate_trace={})
        return OrchestrationResult(final_answer="single ok", safety_status="PASS", debate_trace={})

    monkeypatch.setattr(continue_orch, "_run_strategy", _partial_then_pass)
    monkeypatch.setattr(continue_orch, "_attach_execution_evidence", lambda out, records: None)
    monkeypatch.setattr(continue_orch, "_apply_execution_judge", lambda out, records: None)

    recovered = continue_orch._rollback_on_gate_fail(Strategy.DEBATE, "q", [], None, [], failed)
    assert continued == [Strategy.ENSEMBLE, Strategy.SINGLE]
    assert recovered.final_answer == "single ok"


def test_claim_extraction_and_execution_evidence_from_records():
    orch = _orch()
    text = "Short. Diabetes biomarker association is significant in this synthetic cohort. " * 6
    claims = orch._extract_claims(text, source_model="m1")
    records = [
        _Rec("train_model", {"auc": 0.8, "feature": "diabetes biomarker"}, {"icd10_code": "E11"}),
        _Rec("statistical_review", {"status": "PASS"}),
        _Rec("cohort_summary", {"error": "bad"}),
    ]

    links = orch._evidence_from_records(claims, records)

    assert orch._extract_claims("", source_model="m1") == []
    assert 1 <= len(claims) <= 5
    assert any(link.relation == "supports" for link in links)
    assert any(link.relation == "adjudicates" for link in links)
    assert all("bad" not in link.snippet for link in links)
    assert orch._evidence_from_records([], records) == []
    assert orch._evidence_from_records(claims, []) == []

    no_overlap = orch._evidence_from_records(
        [Claim("c2", "Diabetes biomarker association is significant.")],
        [_Rec("cohort_summary", {"unrelated": "value"}), _Rec("cohort_summary", {"error": "bad"})],
    )
    assert no_overlap == []

    adjudication_only = orch._evidence_from_records(
        [Claim("c3", "Diabetes biomarker association is significant.")],
        [_Rec("safety_check", {"status": "PASS"})],
    )
    assert len(adjudication_only) == 1
    assert adjudication_only[0].relation == "adjudicates"


def test_attach_execution_evidence_deduplicates_links():
    orch = _orch()
    claim = Claim("c1", "Diabetes biomarker association is significant.")
    existing = EvidenceLink("c1", "execution_record", "2026:train_model", "supports")
    out = OrchestrationResult(
        final_answer=claim.text,
        claims=[claim],
        evidence_links=[existing],
        debate_trace={"strategy": "single"},
    )

    monkeypatch_records = [_Rec("train_model", {"diabetes": "biomarker association"})]
    orch._attach_execution_evidence(out, monkeypatch_records)
    orch._attach_execution_evidence(out, monkeypatch_records)

    keys = {(e.claim_id, e.evidence_type, e.evidence_id, e.relation) for e in out.evidence_links}
    assert len(keys) == len(out.evidence_links)
    assert orch._has_execution_evidence(out) is True
    assert orch._is_execution_evidence_link(EvidenceLink("c1", "analysis_record", "r")) is True
    assert orch._is_execution_evidence_link(EvidenceLink("c1", "debate_proposal", "r")) is False

    no_new = OrchestrationResult(final_answer=claim.text, claims=[claim], evidence_links=[])
    orch._attach_execution_evidence(no_new, [_Rec("unrelated", {"x": "y"})])
    assert no_new.evidence_links == []


def test_execution_judge_appends_caution_and_tool_hint_paths():
    orch = _orch()
    caution = OrchestrationResult(
        final_answer="Draft descriptive note.",
        claims=[],
        evidence_links=[],
        debate_trace={"strategy": "single"},
        llm_response=LLMResponse(text="Draft descriptive note."),
    )
    orch._apply_execution_judge(caution, [_Rec("cohort_summary", {"n_cases": 20})])
    assert caution.safety_status == "PARTIAL"
    assert "n_cases >= 100" in caution.text

    blocked = OrchestrationResult(
        final_answer="We conclude causal risk increases.",
        claims=[Claim("c1", "causal risk increases")],
        evidence_links=[],
        debate_trace={"strategy": "debate"},
        llm_response=LLMResponse(
            text="We conclude causal risk increases.",
            tool_calls=[ToolCall(id="t1", name="safety_check", args={})],
        ),
    )
    orch._apply_execution_judge(blocked, [_Rec("cohort_summary", {"n_cases": 200})])
    assert blocked.safety_status == "PARTIAL"
    assert "Pending adjudication tools: safety_check" in blocked.text


def test_debate_fallback_paths_and_already_requested(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2")])
    monkeypatch.setattr(orch, "_parallel_chat", lambda models, messages, tools: [
        DebateProposal("m2", LLMResponse(text="Single proposal answer with enough length."))
    ])
    one = orch.debate("q", [{"role": "user", "content": "q"}])
    assert one.text.startswith("Single proposal")

    fallback_client = _Client([LLMResponse(text="Default fallback answer with enough length.")])
    monkeypatch.setattr(orch, "get_client", lambda _model_id: fallback_client)
    monkeypatch.setattr(orch, "_parallel_chat", lambda models, messages, tools: [])
    none = orch.debate("q", [{"role": "user", "content": "q"}])
    assert none.text.startswith("Default fallback")

    assert orch._adjudication_already_requested([]) is False
    assert orch._adjudication_already_requested([
        {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "safety_check"}}]}
    ]) is True
    assert orch._adjudication_already_requested([
        {"role": "assistant", "tool_calls": [{"id": "adjudicate_abc_1", "name": "other"}]}
    ]) is True
    assert orch._adjudication_already_requested([
        {"role": "assistant", "tool_calls": [object(), {"id": "ordinary", "name": "other"}]}
    ]) is False


def test_debate_successful_critiques_are_attached(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2"), ModelSpec("judge", role="judge")])
    proposals = [
        DebateProposal("m1", LLMResponse(text="Proposal one states a diabetes biomarker association.")),
        DebateProposal("m2", LLMResponse(text="Proposal two states a diabetes biomarker replication.")),
    ]
    clients = {
        "m1": _Client([LLMResponse(text="critique one")]),
        "m2": _Client([LLMResponse(text="critique two")]),
        "judge": _Client([LLMResponse(text="Judge synthesis with enough detail to extract a scientific claim.")]),
    }

    monkeypatch.setattr(orch, "_parallel_chat", lambda models, messages, tools: proposals)
    monkeypatch.setattr(orch, "_anonymous_initial_votes", lambda proposers, props: [{"vote": "A"}, {"vote": "A"}])
    monkeypatch.setattr(orch, "get_client", lambda model_id: clients[model_id])

    out = orch.debate("q", [{"role": "user", "content": "q"}])

    assert out.debate_trace["strategy"] == "debate"
    assert [p.critique for p in proposals] == ["critique one", "critique two"]


def test_debate_disagreement_without_available_adjudication_tools(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2"), ModelSpec("judge", role="judge")])
    proposals = [
        DebateProposal("m1", LLMResponse(text="Proposal one supports a diabetes biomarker.")),
        DebateProposal("m2", LLMResponse(text="Proposal two disputes the same biomarker.")),
    ]
    clients = {
        "m1": _Client([LLMResponse(text="critique one")]),
        "m2": _Client([LLMResponse(text="critique two")]),
        "judge": _Client([LLMResponse(text="Judge synthesis without tool calls.")]),
    }
    monkeypatch.setattr(orch, "_parallel_chat", lambda models, messages, tools: proposals)
    monkeypatch.setattr(orch, "_anonymous_initial_votes", lambda proposers, props: [{"vote": "A"}, {"vote": "B"}])
    monkeypatch.setattr(orch, "_suggest_reretrieval_queries", lambda query, props: ["replication query"])
    monkeypatch.setattr(orch, "get_client", lambda model_id: clients[model_id])

    out = orch.debate("q", [{"role": "user", "content": "q"}], tools=[])

    assert out.debate_trace["disagreement"] is True
    assert out.debate_trace["forced_adjudication"] is False
    assert out.tool_calls == []


def test_build_adjudication_tool_calls_filters_and_limits():
    orch = _orch()
    calls = orch._build_adjudication_tool_calls(
        tools=[
            {"type": "not_function", "function": {"name": "ignored"}},
            {"type": "function", "function": {"name": ""}},
            {"type": "function", "function": {"name": "statistical_review"}},
            {"type": "function", "function": {"name": "safety_check"}},
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "recall_session"}},
            object(),
        ],
        query="q" * 400,
        reretrieval_queries=["x" * 500],
    )

    assert [c.name for c in calls] == ["statistical_review", "safety_check", "web_search"]
    assert calls[2].args["query"] == "x" * 240
    assert orch._build_adjudication_tool_calls(None, "q") == []


def test_votes_reretrieval_judge_and_proposal_evidence(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2"), ModelSpec("judge", role="judge")])
    clients = {
        "m1": _Client(['{"vote":"a","confidence":0.9,"reason":"best"}']),
        "m2": _Client([RuntimeError("vote fail")]),
        "judge": _Client([LLMResponse(text="Judge synthesis.")]),
    }
    monkeypatch.setattr(orch, "get_client", lambda model_id: clients[model_id])
    proposals = [
        DebateProposal("m1", LLMResponse(text="Diabetes biomarker association support."), critique="critique"),
        DebateProposal("m2", LLMResponse(text="Diabetes biomarker alternate support.")),
    ]

    votes = orch._anonymous_initial_votes([ModelSpec("m1"), ModelSpec("m2")], proposals)
    judge = orch._judge_select(ModelSpec("judge", role="judge"), proposals, [{"role": "user", "content": "q"}], None)
    claims = [Claim("c1", "Diabetes biomarker association support.")]
    links = orch._evidence_from_proposals(claims, proposals)

    assert votes[0]["vote"] == "A"
    assert votes[1]["reason"] == "vote_unavailable"
    assert judge.text.startswith("*[Multi-model debate: m1, m2")
    assert len(links) == 2
    assert orch._evidence_from_proposals([Claim("c2", "123 456")], proposals) == []


def test_suggest_reretrieval_queries_parse_and_fallback(monkeypatch):
    orch = _orch([ModelSpec("m1")])
    client = _Client(['```json\n["query one", "query two", "query three", "query four"]\n```'])
    monkeypatch.setattr(orch, "get_client", lambda _model_id: client)

    parsed = orch._suggest_reretrieval_queries("diabetes", [DebateProposal("m1", LLMResponse(text="a"))])
    client.responses = ['{"not":"a list"}']
    non_list = orch._suggest_reretrieval_queries("diabetes", [])
    client.responses = [RuntimeError("offline")]
    fallback = orch._suggest_reretrieval_queries("diabetes", [])

    assert parsed == ["query one", "query two", "query three"]
    assert non_list[0] == "diabetes validation study"
    assert fallback[0] == "diabetes validation study"


def test_parallel_chat_success_and_failure_sorting(monkeypatch):
    orch = _orch([ModelSpec("low", priority=1), ModelSpec("high", priority=10), ModelSpec("bad", priority=99)])
    clients = {
        "low": _Client([LLMResponse(text="low")]),
        "high": _Client([LLMResponse(text="high")]),
        "bad": _Client([RuntimeError("bad")]),
    }
    monkeypatch.setattr(orch, "get_client", lambda model_id: clients[model_id])

    proposals = orch._parallel_chat(orch.model_pool, [{"role": "user", "content": "q"}], None)

    assert [p.model_id for p in proposals] == ["high", "low"]


def test_supervisor_fallback_success_and_specialist_failure(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2"), ModelSpec("judge", role="judge")])
    fallback_client = _Client(["not json", LLMResponse(text="fallback supervisor answer with enough length.")])
    monkeypatch.setattr(orch, "get_client", lambda _model_id: fallback_client)
    fallback = orch.supervisor("q", [{"role": "user", "content": "q"}])
    assert fallback.text.startswith("fallback supervisor")

    clients = {
        "m1": _Client([
            '[{"task":"extract cohort","focus":"phenotype"},{"task":"check stats","focus":"statistics"}]',
            LLMResponse(text="merged answer with enough detail to become a claim."),
            LLMResponse(text="specialist one"),
        ]),
        "m2": _Client([RuntimeError("specialist failed")]),
        "judge": _Client([LLMResponse(text="unused")]),
    }
    monkeypatch.setattr(orch, "get_client", lambda model_id: clients[model_id])
    out = orch.supervisor("q", [{"role": "user", "content": "q"}])
    assert out.debate_trace["strategy"] == "supervisor"
    assert out.debate_trace["subtasks"][0]["task"] == "extract cohort"
    assert len(out.debate_trace["specialist_results"]) == 1


def test_ensemble_paths(monkeypatch):
    orch = _orch([ModelSpec("m1"), ModelSpec("m2")])
    default_client = _Client([
        LLMResponse(text="default ensemble fallback with enough detail."),
        LLMResponse(text="merged ensemble answer with enough detail to form claims."),
    ])
    monkeypatch.setattr(orch, "get_client", lambda _model_id: default_client)

    monkeypatch.setattr(orch, "_parallel_chat", lambda models, messages, tools: [])
    no_props = orch.ensemble([{"role": "user", "content": "q"}])
    assert no_props.text.startswith("default ensemble")

    monkeypatch.setattr(
        orch,
        "_parallel_chat",
        lambda models, messages, tools: [DebateProposal("m2", LLMResponse(text="single prop answer with enough detail."))],
    )
    one = orch.ensemble([{"role": "user", "content": "q"}])
    assert one.debate_trace["strategy"] == "ensemble"

    tool_resp = LLMResponse(text="", tool_calls=[ToolCall("t1", "prevalence", {})])
    monkeypatch.setattr(
        orch,
        "_parallel_chat",
        lambda models, messages, tools: [
            DebateProposal("m1", LLMResponse(text="plain answer with enough detail.")),
            DebateProposal("m2", tool_resp),
        ],
    )
    tool_choice = orch.ensemble([{"role": "user", "content": "q"}])
    assert tool_choice.debate_trace["selection_reason"] == "tool_calls_present"
    assert tool_choice.debate_trace["selected_model"] == "m2"

    monkeypatch.setattr(
        orch,
        "_parallel_chat",
        lambda models, messages, tools: [
            DebateProposal("m1", LLMResponse(text="Diabetes biomarker answer supports association.")),
            DebateProposal("m2", LLMResponse(text="Diabetes biomarker answer supports replication.")),
        ],
    )
    merged = orch.ensemble([{"role": "user", "content": "q"}])
    assert merged.text.startswith("merged ensemble")
    assert merged.debate_trace["participants"] == ["m1", "m2"]


def test_parse_json_and_subagent_dispatch(monkeypatch):
    orch = _orch([ModelSpec("m1")])
    client = _Client([LLMResponse(text="subagent result")])
    monkeypatch.setattr(orch, "get_client", lambda model_id: client)

    assert orch._parse_json_loose("") is None
    assert orch._parse_json_loose("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert orch._parse_json_loose("```text\nnot json\n```") is None
    assert orch._parse_json_loose("not json") is None
    messages = orch._build_self_contained_message(
        SubagentRole.EXPLORER,
        "inspect files",
        {"scope": "biobank_agent/planner.py"},
    )
    no_context_messages = orch._build_self_contained_message(SubagentRole.WORKER, "implement fix")
    result = orch.dispatch_subagent(
        SubagentCall(role=SubagentRole.EXPLORER, task="inspect files"),
        context_pack={"scope": "biobank_agent/planner.py"},
        model_id="m1",
    )

    assert messages[0]["role"] == "system"
    assert "MUST NOT spawn sub-agents" in messages[0]["content"]
    assert "biobank_agent/planner.py" in messages[0]["content"]
    assert "## Context" not in no_context_messages[0]["content"]
    assert result.text == "subagent result"
