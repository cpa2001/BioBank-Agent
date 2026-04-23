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
    decision = rag.analyze("I do not know and need to check latest evidence", context="")
    assert decision.should_retrieve is True
    assert decision.source in ("dual_channel", "web_search")
