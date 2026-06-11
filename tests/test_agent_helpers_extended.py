"""Direct tests for Agent helper methods without running the LLM loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from biobank_agent.agent import Agent
from biobank_agent.orchestrator import ModelSpec
from biobank_agent.state import AnalysisRecord


class RecordingMemory:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.domain = SimpleNamespace(
            summary=lambda max_chars=2000: "domain facts",
            append_finding=self.append_finding,
        )
        self.user = SimpleNamespace(
            summary=lambda: "user profile",
            upsert_preference=self.upsert_preference,
        )
        self.sessions = SimpleNamespace(index_turn=lambda **kwargs: None)
        self.findings = []
        self.preferences = {}
        self.saved_pipelines = {}

    def summary(self):
        return "memory summary"

    def upsert_node(self, node_type, node_id, payload=None, score=0.0):
        self.nodes.append((node_type, node_id, payload or {}, score))

    def link_nodes(self, src_type, src_id, dst_type, dst_id, relation, weight=1.0, evidence=None):
        self.links.append((src_type, src_id, dst_type, dst_id, relation, weight, evidence))

    def append_finding(self, title, body):
        self.findings.append((title, body))

    def list_pipelines(self):
        return list(self.saved_pipelines)

    def save_pipeline(self, name, steps):
        self.saved_pipelines[name] = steps

    def upsert_preference(self, key, value):
        self.preferences[key] = value


class FakeState:
    def __init__(self):
        self.records = []
        self.figures = []
        self.provenances = []
        self.memory = None

    def context_summary(self):
        return "session summary"


def make_agent(tmp_path):
    agent = Agent.__new__(Agent)
    agent.settings = SimpleNamespace(
        biobank_name="UK Biobank",
        bank_id="ukb",
        reports_dir=tmp_path,
        llm_model="gpt-test",
        llm_api_key="key",
        auto_discover_models=True,
        model_pool="",
        preferred_multi_models="gpt-a,gpt-b",
        max_auto_model_pool=3,
    )
    agent.dm = SimpleNamespace(
        count_subjects=lambda: 500_000,
        conn=SimpleNamespace(execute=lambda sql: SimpleNamespace(fetchone=lambda: [42])),
    )
    agent.catalog = SimpleNamespace(fields={"30740": {"title": "Glucose"}})
    agent.state = FakeState()
    agent.memory = RecordingMemory()
    agent.llm = SimpleNamespace(
        model="gpt-test",
        list_models=lambda refresh=False: ["gpt-a", "gpt-b", "gpt-c"],
    )
    agent.orchestrator = SimpleNamespace(default_model="gpt-test", model_pool=[])
    agent.multimodal_grounder = SimpleNamespace(
        figure_to_tensor=lambda fig: [[1.0, 2.0]],
        extract_structural_signals=lambda table=None, modality_tensors=None: [
            SimpleNamespace(modality="figure", signal_type="variance_proxy", value=0.5, detail="std")
        ],
    )
    agent._active_query_id = "q1"
    agent._review_lock = __import__("threading").Lock()
    agent._last_orchestration_strategy = "single"
    return agent


def rec(skill, args=None, key_results=None, figure_paths=None):
    return AnalysisRecord(
        timestamp="2026-01-01T00:00:00",
        skill=skill,
        args=args or {},
        key_results=key_results or {},
        figure_paths=figure_paths or [],
    )


def test_agent_data_description_system_message_model_refresh_and_switch(tmp_path):
    agent = make_agent(tmp_path)

    description = agent._build_data_description()
    system = agent._system_message()
    refreshed = agent.refresh_available_models()
    agent.switch_model("gpt-new")
    agent.switch_model(" ")

    assert "500,000 participants" in description
    assert "42 diagnosis records" in description
    assert "42 death records" in description
    assert "1 field definitions" in description
    assert system["role"] == "system"
    assert "Long-term Memory" in system["content"]
    assert "Domain Knowledge" in system["content"]
    assert "Researcher Profile" in system["content"]
    assert refreshed == ["gpt-a", "gpt-b", "gpt-c"]
    assert agent.llm.model == "gpt-new"
    assert agent.orchestrator.default_model == "gpt-new"
    assert agent.orchestrator.model_pool[0].model_id == "gpt-new"

    agent.settings.llm_api_key = "your-key"
    assert agent.refresh_available_models() == ["gpt-a", "gpt-b", "gpt-c"]

    agent.settings.llm_api_key = "key"
    agent.settings.model_pool = "explicit"
    agent.orchestrator.model_pool = [ModelSpec("keep")]
    assert agent.refresh_available_models() == ["gpt-a", "gpt-b", "gpt-c"]
    assert [spec.model_id for spec in agent.orchestrator.model_pool] == ["keep"]

    agent.switch_model("keep")
    assert [spec.model_id for spec in agent.orchestrator.model_pool] == ["keep"]

    failing = make_agent(tmp_path)
    failing.dm = SimpleNamespace(
        count_subjects=lambda: (_ for _ in ()).throw(RuntimeError("no data")),
        conn=SimpleNamespace(execute=lambda sql: (_ for _ in ()).throw(RuntimeError("no table"))),
    )
    failing.catalog = SimpleNamespace(fields={})
    assert "Participant data available" in failing._build_data_description()

    class BrokenCatalog:
        @property
        def fields(self):
            raise RuntimeError("catalog unavailable")

    quiet = make_agent(tmp_path)
    quiet.catalog = BrokenCatalog()
    quiet.memory = SimpleNamespace(
        summary=lambda: "",
        domain=SimpleNamespace(summary=lambda max_chars=2000: ""),
        user=SimpleNamespace(summary=lambda: ""),
    )
    quiet_system = quiet._system_message()
    assert "Long-term Memory" not in quiet_system["content"]
    assert "Domain Knowledge" not in quiet_system["content"]
    assert "Researcher Profile" not in quiet_system["content"]

    settings = SimpleNamespace(
        llm_model="base",
        model_pool="base, alt,base",
        preferred_multi_models="",
        max_auto_model_pool=3,
    )
    assert [spec.model_id for spec in Agent._build_model_pool(settings)] == ["base", "alt"]


def test_agent_orchestration_and_tool_graph_recording(tmp_path):
    agent = make_agent(tmp_path)
    orch = SimpleNamespace(
        claims=[
            SimpleNamespace(
                claim_id="c1",
                text="Claim text",
                source_model="gpt-a",
                confidence=0.8,
                tags=["biomarker"],
            )
        ],
        evidence_links=[
            SimpleNamespace(
                evidence_type="result",
                evidence_id="r1",
                snippet="evidence",
                source_model="gpt-b",
                score=0.7,
                claim_id="c1",
                relation="supports",
            )
        ],
        debate_trace={"strategy": "ensemble"},
        safety_status="PASS",
    )

    agent._record_orchestration_graph(orch)
    agent._record_tool_graph(
        skill="phewas",
        args={"field_id": "30740", "note": "abc"},
        key_results={"n_cases": 123, "p_value": 0.001, "summary": "hit", "error": "ignored"},
        figure_paths=[str(tmp_path / "fig.png")],
        timestamp="2026-01-01T00:00:00",
    )

    node_types = [n[0] for n in agent.memory.nodes]
    relations = [l[4] for l in agent.memory.links]

    assert "claim" in node_types
    assert "result" in node_types
    assert "tool" in node_types
    assert "field" in node_types
    assert "figure" in node_types
    assert "signal" in node_types
    assert "produced_claim" in relations
    assert "has_orchestration_trace" in relations
    assert "uses_field" in relations
    assert "has_structure_signal" in relations
    assert "derived_claim" in relations

    inactive = make_agent(tmp_path)
    inactive._active_query_id = None
    inactive._record_orchestration_graph(orch)
    inactive._record_tool_graph("skill", {}, {}, [], "ts")
    assert inactive.memory.nodes == []


def test_agent_tool_graph_ignores_multimodal_grounding_errors(tmp_path):
    agent = make_agent(tmp_path)
    agent.multimodal_grounder = SimpleNamespace(
        figure_to_tensor=lambda fig: (_ for _ in ()).throw(RuntimeError("cannot decode")),
        extract_structural_signals=lambda **kwargs: [],
    )

    agent._record_tool_graph(
        skill="plot",
        args={},
        key_results={"status": "ok"},
        figure_paths=[str(tmp_path / "broken.png")],
        timestamp="2026-01-01T00:00:00",
    )

    assert "figure" in [n[0] for n in agent.memory.nodes]
    assert "signal" not in [n[0] for n in agent.memory.nodes]


def test_agent_background_review_domain_findings_patterns_profile_and_ctx(tmp_path):
    agent = make_agent(tmp_path)
    agent.state.records = [
        rec("train_model", {"icd10_code": "E11", "model_type": "xgb"}, {"mean_auc": 0.80, "top_features": ["Glucose"]}),
        rec("survival", {"icd10_code": "E11"}, {"log_rank_p": 0.001}),
        rec("think"),
        rec("prevalence", {"icd10_code": "E11"}),
        rec("cohort_summary", {"icd10_code": "E11"}),
        rec("train_model", {"icd10_code": "E11", "model_type": "xgb"}, {"mean_auc": 0.97}),
        rec("prevalence", {"icd10_code": "I10"}),
        rec("cohort_summary", {"icd10_code": "I10"}),
        rec("train_model", {"icd10_code": "I10", "model_type": "xgb"}, {"mean_auc": 0.75}),
    ]

    agent._save_domain_findings(agent.state.records)
    agent._detect_pipeline_patterns()
    agent._update_user_profile()
    ctx = agent._build_ctx(tmp_path / "custom_report")

    assert any(title == "Predictor: E11" for title, _ in agent.memory.findings)
    assert any(title == "WARNING: Possible leakage — E11" for title, _ in agent.memory.findings)
    assert any(title == "Survival: E11" for title, _ in agent.memory.findings)
    assert "prevalence_cohort_summary_train_model" in agent.memory.saved_pipelines
    assert agent.memory.preferences["commonly_studied_diseases"].startswith("E11")
    assert agent.memory.preferences["preferred_model"] == "xgb"
    assert ctx.dm is agent.dm
    assert ctx.catalog is agent.catalog
    assert ctx.state.memory is agent.memory
    assert ctx.report_dir == tmp_path / "custom_report"

    agent._background_review()
    assert agent._review_lock.acquire(blocking=False)
    agent._review_lock.release()


def test_agent_background_review_noop_and_failure_paths(tmp_path):
    agent = make_agent(tmp_path)
    agent._review_lock = SimpleNamespace(acquire=lambda blocking=False: False, release=lambda: (_ for _ in ()).throw(AssertionError("no release")))
    agent._background_review()

    quiet = make_agent(tmp_path)
    quiet.state.records = [
        rec("train_model", {"icd10_code": "E11"}, {"mean_auc": 0.60}),
        rec("survival", {"icd10_code": "E11"}, {"log_rank_p": 0.2}),
        rec("prevalence"),
    ]
    quiet._save_domain_findings(quiet.state.records)
    quiet._detect_pipeline_patterns()
    quiet._update_user_profile()
    assert quiet.memory.findings == []
    assert quiet.memory.saved_pipelines == {}
    assert quiet.memory.preferences["commonly_studied_diseases"].startswith("E11")
    assert "preferred_model" not in quiet.memory.preferences

    model_only = make_agent(tmp_path)
    model_only.state.records = [
        rec("train_model", {"model_type": "logistic"}, {"mean_auc": 0.70}),
    ]
    model_only._update_user_profile()
    assert "commonly_studied_diseases" not in model_only.memory.preferences
    assert model_only.memory.preferences["preferred_model"] == "logistic"

    existing = make_agent(tmp_path)
    existing.state.records = [
        rec("a"), rec("b"), rec("c"),
        rec("a"), rec("b"), rec("c"),
    ]
    existing.memory.saved_pipelines["a_b_c"] = [{"skill": "a", "args": {}}]
    existing._detect_pipeline_patterns()
    assert existing.memory.saved_pipelines == {"a_b_c": [{"skill": "a", "args": {}}]}

    class Lock:
        def __init__(self):
            self.released = False

        def acquire(self, blocking=False):
            return True

        def release(self):
            self.released = True

    failing = make_agent(tmp_path)
    lock = Lock()
    failing._review_lock = lock
    failing._save_domain_findings = lambda records: (_ for _ in ()).throw(RuntimeError("review failed"))
    failing._background_review()
    assert lock.released is True
