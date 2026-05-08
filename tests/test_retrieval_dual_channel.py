"""Tests for dual-channel retrieval + sufficiency reranking."""

from types import SimpleNamespace

from biobank_agent.retrieval import AgenticRAG


def test_dual_channel_retrieve_merges_semantic_and_graph():
    memory = SimpleNamespace()
    memory.sessions = SimpleNamespace(
        search=lambda query, limit=4: [
            {"session_id": "s1", "user_query": "diabetes risk", "key_findings": "n_cases=300"},
        ]
    )
    memory.action_graph = SimpleNamespace(
        search_nodes=lambda query, node_types=None, limit=4: [
            {"node_type": "claim", "node_id": "c1", "payload": {"text": "diabetes risk high", "n_cases": 500}},
        ]
    )
    rag = AgenticRAG(llm=None, memory=memory, threshold=0.1)
    hits = rag.dual_channel_retrieve("diabetes risk", limit=6)
    assert len(hits) >= 2
    channels = {h["channel"] for h in hits}
    assert "semantic" in channels
    assert "graph" in channels


def test_analyze_returns_dual_channel_hits_on_explicit_gap():
    memory = SimpleNamespace()
    memory.sessions = SimpleNamespace(search=lambda query, limit=4: [])
    memory.action_graph = SimpleNamespace(search_nodes=lambda query, node_types=None, limit=4: [])
    rag = AgenticRAG(llm=None, memory=memory, threshold=0.1)
    decision = rag.analyze("I do not know the prior diabetes session result", context="")
    assert decision.should_retrieve is True
    assert decision.source == "dual_channel"
    assert decision.channels == ["semantic", "graph"]
    assert decision.context_hits == []


def test_analyze_fast_paths_threshold_and_query_routing():
    rag = AgenticRAG(llm=None, memory=None, threshold=0.6)

    empty = rag.analyze("short")
    below = rag.analyze("BRCA1 biomarker", context="")
    academic = rag.analyze("Need a reference citation paper for Diabetes prediction study", context="")
    temporal = rag.analyze("Need the latest 2025 update on UK Biobank assay guidance", context="")
    default = rag.analyze("This is unclear for biomarker interpretation", context="")

    assert empty.should_retrieve is False
    assert below.should_retrieve is False
    assert below.reason == "Below threshold (0.20 < 0.6)"
    assert academic.source == "web_search"
    assert academic.query.endswith("biomedical study")
    assert temporal.source == "web_search"
    assert temporal.query.endswith("2025 2026")
    assert default.should_retrieve is True
    assert default.source == "web_search"
    assert default.channels == ["semantic"]


def test_dual_channel_retrieve_handles_missing_memory_errors_and_empty_hits():
    assert AgenticRAG(memory=None).dual_channel_retrieve("diabetes") == []

    assert AgenticRAG(memory=SimpleNamespace(sessions=SimpleNamespace(search=lambda *a, **k: []))).dual_channel_retrieve(
        "diabetes"
    ) == []
    assert AgenticRAG(
        memory=SimpleNamespace(action_graph=SimpleNamespace(search_nodes=lambda *a, **k: []))
    ).dual_channel_retrieve("diabetes") == []

    memory = SimpleNamespace(
        sessions=SimpleNamespace(search=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("semantic down"))),
        action_graph=SimpleNamespace(search_nodes=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("graph down"))),
    )
    assert AgenticRAG(memory=memory).dual_channel_retrieve("diabetes") == []

    no_hits = SimpleNamespace(
        sessions=SimpleNamespace(search=lambda *args, **kwargs: []),
        action_graph=SimpleNamespace(search_nodes=lambda *args, **kwargs: []),
    )
    assert AgenticRAG(memory=no_hits).dual_channel_retrieve("diabetes") == []


def test_dual_channel_scoring_sufficiency_and_non_dict_graph_hits():
    memory = SimpleNamespace(
        sessions=SimpleNamespace(
            search=lambda *args, **kwargs: [
                {"session_id": "weak", "user_query": "diabetes", "key_findings": "", "n_cases": 99},
                {"session_id": "strong", "user_query": "diabetes risk", "key_findings": "validated", "n_controls": 150},
            ]
        ),
        action_graph=SimpleNamespace(
            search_nodes=lambda *args, **kwargs: [
                {"node_type": "result", "node_id": "r1", "payload": {"title": "diabetes risk", "payload": {"n": 1}}},
            ]
        ),
    )
    rag = AgenticRAG(memory=memory)

    hits = rag.dual_channel_retrieve("diabetes risk", limit=2)

    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
    assert all("sufficiency" in hit for hit in hits)
    assert rag._statistical_sufficiency(["not", "dict"]) == 0.0
    assert rag._statistical_sufficiency({}) == 0.3
    assert rag._statistical_sufficiency({"n_cases": 100, "key_findings": "ok", "payload": {"x": 1}}) == 1.0
    assert rag._statistical_sufficiency({"payload": []}) == 0.0


def test_query_extractors_fallback_to_raw_text():
    rag = AgenticRAG()

    assert rag._extract_academic_query("reference paper needed") == "reference paper needed research paper"
    assert rag._extract_key_terms("the and or but") == "the and or but"
