"""Deterministic tests for Agent.run control-flow edges."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from biobank_agent.agent import Agent
from biobank_agent.llm import LLMResponse, ToolCall
from biobank_agent.orchestrator import Claim, ModelSpec, OrchestrationResult
from biobank_agent.state import SessionState


class _Memory:
    def __init__(self, fail_upsert=False):
        self.fail_upsert = fail_upsert
        self.nodes = []
        self.errors = []
        self.field_usage = []
        self.indexed = []
        self.suggestions = []
        self.sessions = SimpleNamespace(index_turn=self._index_turn)

    def upsert_node(self, *args, **kwargs):
        if self.fail_upsert:
            raise RuntimeError("memory offline")
        self.nodes.append((args, kwargs))

    def _index_turn(self, **kwargs):
        self.indexed.append(kwargs)

    def record_error(self, **kwargs):
        self.errors.append(kwargs)

    def get_error_suggestions(self, *_args):
        return list(self.suggestions)

    def record_field_usage(self, field_id):
        self.field_usage.append(field_id)


class _Registry:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def tool_schemas(self):
        return [{"type": "function", "function": {"name": "fake_skill"}}]

    def execute(self, name, args, ctx=None):
        self.calls.append((name, dict(args), ctx))
        item = self.responses.pop(0) if self.responses else {"ok": True}
        if isinstance(item, Exception):
            raise item
        if item == "__add_figure__":
            ctx.state.figures.append(Path(ctx.report_dir) / "figure.svg")
            return {"n": 1, "status": "ok"}
        return item


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "m1"

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


class _Orchestrator:
    def __init__(self):
        self.model_pool = [ModelSpec("m1"), ModelSpec("m2")]
        self.default_model = "m1"
        self.route_calls = []
        self.wrap_calls = []
        self.attach_calls = 0
        self.judge_calls = 0

    def route(self, **kwargs):
        self.route_calls.append(kwargs)
        raw = LLMResponse(text="routed final", usage={"prompt_tokens": 3})
        return OrchestrationResult(
            final_answer=raw.text,
            claims=[Claim("c1", "Routed answer has enough content for tracing.")],
            debate_trace={"strategy": "supervisor"},
            llm_response=raw,
        )

    def _wrap_llm_response(self, raw, strategy, model_id):
        self.wrap_calls.append((raw, strategy, model_id))
        return OrchestrationResult(
            final_answer=raw.text,
            claims=[Claim("c1", raw.text)] if raw.text else [],
            debate_trace={"strategy": strategy.value},
            llm_response=raw,
        )

    def _attach_execution_evidence(self, *_args, **_kwargs):
        self.attach_calls += 1

    def _apply_execution_judge(self, *_args, **_kwargs):
        self.judge_calls += 1


class _ToolLearner:
    def __init__(self):
        self.records = []

    def record(self, *args):
        self.records.append(args)


class _Repro:
    def __init__(self):
        self.logs = []

    def create_audit_log(self, **kwargs):
        self.logs.append(kwargs)


def _make_agent(tmp_path, llm_responses=None, registry=None):
    agent = Agent.__new__(Agent)
    agent.settings = SimpleNamespace(
        reports_dir=tmp_path,
        max_tool_rounds=3,
        multi_model_enabled=False,
        llm_model="m1",
        bank_id="ukb",
        auto_discover_models=True,
        model_pool="",
        preferred_multi_models="m2,m3",
        max_auto_model_pool=3,
    )
    agent.messages = []
    agent.state = SessionState()
    agent.state.duckdb_conn = object()
    agent.memory = _Memory()
    agent.llm = _LLM(llm_responses or [])
    agent.orchestrator = _Orchestrator()
    agent.registry = registry or _Registry()
    agent.reflexion = None
    agent.tool_learner = _ToolLearner()
    agent.verdict_engine = SimpleNamespace(
        verify_skill_result=lambda *args, **kwargs: SimpleNamespace(n_blockers=0, summary=lambda: "PASS")
    )
    agent.reproducibility = _Repro()
    agent.dm = SimpleNamespace()
    agent.catalog = SimpleNamespace()
    agent._study_spec_compiler = SimpleNamespace(compile=lambda _query: (_ for _ in ()).throw(RuntimeError("skip")))
    agent._current_study_spec = None
    agent._active_query_id = None
    agent._last_orchestration_strategy = "single"
    agent._review_interval = 99
    agent._review_lock = SimpleNamespace(acquire=lambda blocking=False: True, release=lambda: None)
    agent._system_message = lambda: {"role": "system", "content": "system"}
    agent._post_run_calls = []
    agent._post_run = lambda query: agent._post_run_calls.append(query)
    agent._graph_records = []
    agent._record_orchestration_graph = lambda result: agent._graph_records.append(result)
    agent._tool_graph_records = []
    agent._record_tool_graph = lambda **kwargs: agent._tool_graph_records.append(kwargs)
    return agent


def test_agent_init_wires_components_with_patched_dependencies(monkeypatch, tmp_path):
    calls = []

    class FakeDM:
        def __init__(self, settings):
            self.settings = settings
            self.conn = object()

    class FakeCatalog:
        def __init__(self, field_txt, category_txt):
            self.field_txt = field_txt
            self.category_txt = category_txt
            self.fields = {}

    class FakeLLMClient:
        def __init__(self, base_url, api_key, model):
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self.tool_call_content_mode = None

        def list_models(self, refresh=False):
            calls.append(("list_models", refresh))
            return ["m2", "m3"]

    class FakeRegistry:
        def __len__(self):
            return 0

        def list_skills(self):
            return []

    registry = FakeRegistry()
    monkeypatch.setattr("biobank_agent.agent.DataManager", FakeDM)
    monkeypatch.setattr("biobank_agent.agent.FieldCatalog", FakeCatalog)
    monkeypatch.setattr("biobank_agent.agent.LongTermMemory", lambda memory_dir: _Memory())
    monkeypatch.setattr("biobank_agent.agent.LLMClient", FakeLLMClient)
    monkeypatch.setattr("biobank_agent.agent.ReflexionEngine", lambda **kwargs: SimpleNamespace(kind="reflexion"))
    monkeypatch.setattr("biobank_agent.agent.VerdictEngine", lambda llm: SimpleNamespace(kind="verdict"))
    monkeypatch.setattr("biobank_agent.agent.AgenticRAG", lambda **kwargs: SimpleNamespace(kind="rag"))
    monkeypatch.setattr("biobank_agent.agent.ToolLearner", lambda memory: SimpleNamespace(kind="learner"))
    monkeypatch.setattr("biobank_agent.agent.autodiscover_skills", lambda: calls.append(("autodiscover",)))
    monkeypatch.setattr("biobank_agent.agent.discover_custom_skills", lambda custom_dir: calls.append(("custom", custom_dir)))
    monkeypatch.setattr("biobank_agent.agent.get_registry", lambda: registry)
    monkeypatch.setattr("biobank_agent.difficulty.DifficultyEstimator", lambda history_path: SimpleNamespace(history_path=history_path))
    monkeypatch.setattr("biobank_agent.reproducibility.ReproducibilityHarness", lambda checkpoint_dir: SimpleNamespace(checkpoint_dir=checkpoint_dir))
    monkeypatch.setattr("biobank_agent.study_spec.StudySpecCompiler", lambda llm: SimpleNamespace(llm=llm))

    settings = SimpleNamespace(
        ensure_dirs=lambda: calls.append(("ensure_dirs",)),
        field_txt=tmp_path / "field.txt",
        category_txt=tmp_path / "category.txt",
        memory_dir=tmp_path / "memory",
        reports_dir=tmp_path / "reports",
        custom_skills_dir=tmp_path / "skills",
        llm_base_url="http://llm",
        llm_api_key="key",
        llm_model="m1",
        tool_call_content_mode="empty",
        auto_discover_models=True,
        model_pool="",
        preferred_multi_models="m2,m3",
        max_auto_model_pool=3,
        complexity_threshold=0.7,
        debate_rounds=2,
        enable_reflexion=True,
    )

    agent = Agent(settings)

    assert ("ensure_dirs",) in calls
    assert ("list_models", False) in calls
    assert ("autodiscover",) in calls
    assert ("custom", settings.custom_skills_dir) in calls
    assert agent.dm.settings is settings
    assert agent.catalog.field_txt == settings.field_txt
    assert agent.state.duckdb_conn is agent.dm.conn
    assert agent.llm.tool_call_content_mode == "empty"
    assert [spec.model_id for spec in agent.orchestrator.model_pool] == ["m1", "m2", "m3"]
    assert agent.registry is registry
    assert agent.messages == []
    assert agent._review_interval == 5


def test_model_discovery_startup_guards(tmp_path):
    agent = _make_agent(tmp_path)

    agent.settings.auto_discover_models = False
    assert agent._discover_available_models() == []

    agent.settings.auto_discover_models = True
    agent.settings.llm_api_key = "your-key"
    assert agent._discover_available_models() == []

    agent.settings.llm_api_key = "key"
    agent.llm.list_models = lambda refresh=False: (_ for _ in ()).throw(RuntimeError("offline"))
    assert agent._discover_available_models() == []

    agent.available_models = []
    agent.refresh_available_models()
    assert agent.available_models == []


def test_run_single_model_final_answer_updates_trace_tokens_and_messages(tmp_path):
    agent = _make_agent(
        tmp_path,
        [LLMResponse(text="final answer", usage={"prompt_tokens": 2, "completion_tokens": 4})],
    )
    agent.memory = _Memory(fail_upsert=True)

    out = agent.run("hello")

    assert out == "final answer"
    assert agent.messages[-1] == {"role": "assistant", "content": "final answer"}
    assert agent.state.token_usage.total_tokens == 6
    assert agent.state.last_orchestration["final_answer"] == "final answer"
    assert agent._post_run_calls == ["hello"]
    assert len(agent._graph_records) == 1
    assert agent.orchestrator.attach_calls == 1
    assert agent.orchestrator.judge_calls == 1


def test_run_compile_spec_and_ignores_execution_evidence_and_graph_errors(tmp_path):
    agent = _make_agent(
        tmp_path,
        [LLMResponse(text="final answer")],
    )
    agent._study_spec_compiler = SimpleNamespace(
        compile=lambda _query: SimpleNamespace(
            design=SimpleNamespace(value="case_control"),
            modalities=[SimpleNamespace(value="tabular")],
            tool_budget=3,
        )
    )
    agent.orchestrator._attach_execution_evidence = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("attach failed"))
    agent._record_orchestration_graph = lambda result: (_ for _ in ()).throw(RuntimeError("graph failed"))

    out = agent.run("compile spec")

    assert out == "final answer"
    assert agent._current_study_spec.tool_budget == 3


def test_run_handles_empty_and_failing_orchestration_trace_payloads(tmp_path):
    class EmptyTraceResult:
        text = "empty trace final"
        usage = {}
        has_tool_calls = False
        tool_calls = []

        def to_dict(self):
            return {}

    class RaisingTraceResult:
        text = "raising trace final"
        usage = {}
        has_tool_calls = False
        tool_calls = []

        def to_dict(self):
            raise RuntimeError("trace broken")

    empty = _make_agent(tmp_path, [LLMResponse(text="unused")])
    empty.orchestrator._wrap_llm_response = lambda **kwargs: EmptyTraceResult()
    assert empty.run("empty trace") == "empty trace final"
    assert empty.state.last_orchestration == {}

    raising = _make_agent(tmp_path, [LLMResponse(text="unused")])
    raising.orchestrator._wrap_llm_response = lambda **kwargs: RaisingTraceResult()
    assert raising.run("raising trace") == "raising trace final"
    assert raising.state.last_orchestration == {}


def test_run_uses_multi_model_route_with_plan_forced_strategy(tmp_path):
    agent = _make_agent(tmp_path)
    agent.settings.multi_model_enabled = True

    out = agent.run("[PLAN MODE - Status: INTAKE]\nPlan cardiovascular report")

    assert out == "routed final"
    assert agent.orchestrator.route_calls
    assert agent.orchestrator.route_calls[0]["force_strategy"].value == "supervisor"
    assert agent._last_orchestration_strategy == "supervisor"


def test_run_returns_interrupted_before_llm_call(tmp_path):
    agent = _make_agent(tmp_path, [LLMResponse(text="should not be used")])
    agent.state.interrupted = True

    out = agent.run("stop")

    assert out == "[Interrupted by user]"
    assert agent.llm.calls == []


def test_run_executes_tool_records_provenance_and_then_final_answer(tmp_path):
    tool_call = ToolCall(id="tc1", name="fake_skill", args={"field_id": "30740", "top_n": 4})
    registry = _Registry(["__add_figure__"])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call], usage={"prompt_tokens": 1}),
            LLMResponse(text="done", usage={"completion_tokens": 2}),
        ],
        registry=registry,
    )

    out = agent.run("use a tool")

    assert out == "done"
    assert registry.calls[0][0] == "fake_skill"
    assert agent.state.records[0].skill == "fake_skill"
    assert agent.state.records[0].figure_paths[0].endswith("figure.svg")
    assert agent.state.provenances[0].skill == "fake_skill"
    assert agent.memory.field_usage == ["30740"]
    assert agent.tool_learner.records
    assert agent.reproducibility.logs[0]["status"] == "success"
    assert agent.messages[-2]["role"] == "tool"
    assert agent._tool_graph_records[0]["skill"] == "fake_skill"


def test_run_retries_retryable_tool_error_with_reduced_parameters(tmp_path):
    tool_call = ToolCall(
        id="tc1",
        name="retry_skill",
        args={"n_folds": 6, "top_n": 20, "sample_size": 100},
    )
    registry = _Registry([RuntimeError("temporary resource issue"), {"ok": True}])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )

    out = agent.run("retry")

    assert out == "done"
    assert len(registry.calls) == 2
    retry_args = registry.calls[1][1]
    assert retry_args["n_folds"] == 3
    assert retry_args["top_n"] == 10
    assert retry_args["sample_size"] == 100
    assert agent.state.records[0].key_results["_retried_with"] == {
        "n_folds": 3,
        "top_n": 10,
    }


def test_run_reflexion_retry_adds_structured_metadata(tmp_path):
    tool_call = ToolCall(id="tc1", name="reflexive_skill", args={"field_id": "30740"})
    registry = _Registry([ValueError("bad args"), {"ok": True}])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )
    correction = SimpleNamespace(param="field_id", new_value="30750")
    reflection = SimpleNamespace(
        retry_recommended=True,
        corrections=[correction],
        corrected_args={"field_id": "30750"},
        root_cause="wrong field",
        confidence=0.8,
    )
    agent.reflexion = SimpleNamespace(
        should_retry=lambda error, skill_name: True,
        reflect=lambda **kwargs: reflection,
    )

    out = agent.run("reflexion retry")

    assert out == "done"
    assert registry.calls[1][1]["field_id"] == "30750"
    assert agent.state.records[0].key_results["_reflexion"]["corrections"] == {"field_id": "30750"}


def test_run_reflexion_declines_retry_records_original_error(tmp_path):
    tool_call = ToolCall(id="tc1", name="reflexive_skill", args={"field_id": "30740"})
    registry = _Registry([ValueError("bad args")])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )
    agent.reflexion = SimpleNamespace(
        should_retry=lambda error, skill_name: True,
        reflect=lambda **kwargs: SimpleNamespace(retry_recommended=False, corrections=[]),
    )

    assert agent.run("reflexion no retry") == "done"
    assert len(registry.calls) == 1
    assert agent.state.records[0].key_results["error"] == "bad args"


def test_run_reflexion_retry_handles_nondict_result(tmp_path):
    tool_call = ToolCall(id="tc1", name="reflexive_skill", args={"field_id": "30740"})
    registry = _Registry([ValueError("bad args"), "plain retry result"])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )
    agent.reflexion = SimpleNamespace(
        should_retry=lambda error, skill_name: True,
        reflect=lambda **kwargs: SimpleNamespace(
            retry_recommended=True,
            corrections=[SimpleNamespace(param="field_id", new_value="30750")],
            corrected_args={"field_id": "30750"},
            root_cause="wrong field",
            confidence=0.8,
        ),
    )

    assert agent.run("reflexion plain retry") == "done"
    assert agent.state.records[0].key_results == {"result": "plain retry result"}


def test_run_retry_failure_and_error_tracking_failure_are_non_fatal(tmp_path):
    tool_call = ToolCall(id="tc1", name="retry_skill", args={"n_folds": 6})
    registry = _Registry([RuntimeError("temporary resource issue"), RuntimeError("retry failed")])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )
    agent.memory.record_error = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("memory write failed"))

    out = agent.run("retry fails")

    assert out == "done"
    assert len(registry.calls) == 2
    assert agent.state.records[0].key_results == {"error": "temporary resource issue"}


def test_run_legacy_retry_handles_nondict_result(tmp_path):
    tool_call = ToolCall(id="tc1", name="retry_skill", args={"n_folds": 6})
    registry = _Registry([RuntimeError("temporary resource issue"), "plain retry result"])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )

    assert agent.run("legacy plain retry") == "done"
    assert agent.state.records[0].key_results == {"result": "plain retry result"}


def test_run_existing_reflexion_can_skip_retry_without_legacy_fallback(tmp_path):
    tool_call = ToolCall(id="tc1", name="retry_skill", args={"n_folds": 6})
    registry = _Registry([RuntimeError("temporary resource issue")])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )
    agent.reflexion = SimpleNamespace(should_retry=lambda error, skill_name: False)

    assert agent.run("reflexion skip") == "done"
    assert len(registry.calls) == 1
    assert agent.state.records[0].key_results["error"] == "temporary resource issue"


def test_run_records_non_retryable_tool_error_with_known_fixes(tmp_path):
    tool_call = ToolCall(id="tc1", name="bad_skill", args={"field_id": "bad"})
    registry = _Registry([ValueError("invalid field")])
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=registry,
    )
    agent.memory.suggestions = ["Use a valid field ID"]

    out = agent.run("bad")

    assert out == "done"
    assert agent.memory.errors[0]["skill_name"] == "bad_skill"
    assert agent.state.records[0].key_results["error"] == "invalid field"
    assert agent.state.records[0].key_results["known_fixes"] == ["Use a valid field ID"]
    assert agent.state.provenances == []
    assert agent.reproducibility.logs[0]["status"] == "failed"


def test_run_successful_tool_ignores_optional_observer_failures(tmp_path):
    tool_call = ToolCall(id="tc1", name="observed_skill", args={"field_id": "30740"})
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=_Registry([{"ok": True}]),
    )
    agent.tool_learner.record = lambda *args: (_ for _ in ()).throw(RuntimeError("learner down"))
    agent.verdict_engine = SimpleNamespace(
        verify_skill_result=lambda *args, **kwargs: SimpleNamespace(n_blockers=1, summary=lambda: "FAIL")
    )
    agent.reproducibility.create_audit_log = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("audit down"))
    agent._record_tool_graph = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("graph down"))

    out = agent.run("observer failures")

    assert out == "done"
    assert agent.state.records[0].key_results == {"ok": True}
    assert agent.state.provenances[0].skill == "observed_skill"


def test_run_successful_tool_ignores_verdict_exception(tmp_path):
    tool_call = ToolCall(id="tc1", name="verdict_skill", args={})
    agent = _make_agent(
        tmp_path,
        [
            LLMResponse(text="", tool_calls=[tool_call]),
            LLMResponse(text="done"),
        ],
        registry=_Registry([{"ok": True}]),
    )
    agent.verdict_engine = SimpleNamespace(
        verify_skill_result=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verdict down"))
    )

    assert agent.run("verdict exception") == "done"
    assert agent.state.records[0].key_results == {"ok": True}


def test_run_max_tool_rounds_returns_guard_message(tmp_path):
    tool_call = ToolCall(id="tc1", name="fake_skill", args={})
    agent = _make_agent(tmp_path, [LLMResponse(text="", tool_calls=[tool_call])], registry=_Registry([{"ok": True}]))
    agent.settings.max_tool_rounds = 1

    out = agent.run("loop")

    assert out == "[Max tool rounds reached]"
    assert agent.messages[-1] == {"role": "assistant", "content": "[Max tool rounds reached]"}
    assert agent._post_run_calls == ["loop"]


def test_post_run_indexes_outcome_and_triggers_background_review(monkeypatch, tmp_path):
    agent = _make_agent(tmp_path)
    agent._post_run = Agent._post_run.__get__(agent, Agent)
    agent.state.records = [
        SimpleNamespace(timestamp="2026-05-08T00:00:00", key_results={})
        for _ in range(5)
    ]
    agent._review_interval = 5
    agent.difficulty_estimator = SimpleNamespace(calls=[], record_outcome=lambda **kw: agent.difficulty_estimator.calls.append(kw))
    started = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.daemon))

    monkeypatch.setattr("biobank_agent.agent.threading.Thread", FakeThread)

    agent._post_run("query")

    assert agent.difficulty_estimator.calls[0]["query"] == "query"
    assert agent.difficulty_estimator.calls[0]["succeeded"] is True
    assert agent.memory.indexed[0]["user_query"] == "query"
    assert started and started[0][1] is True


def test_post_run_ignores_outcome_and_indexing_failures_without_review(tmp_path):
    agent = _make_agent(tmp_path)
    agent._post_run = Agent._post_run.__get__(agent, Agent)
    agent.state.records = [
        SimpleNamespace(timestamp="2026-05-08T00:00:00", key_results={"error": "bad"})
        for _ in range(4)
    ]
    agent._review_interval = 5
    agent.difficulty_estimator = SimpleNamespace(
        record_outcome=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("estimator down"))
    )
    agent.memory.sessions.index_turn = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("index down"))

    agent._post_run("query")

    assert len(agent.state.records) == 4
