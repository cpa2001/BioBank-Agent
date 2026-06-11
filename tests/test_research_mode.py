"""Tests for unified research mode: RuntimeResearcher + /research."""

from __future__ import annotations

import pytest

from biobank_agent.runtime.council import CouncilError
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.researcher import RuntimeResearcher
from biobank_agent.runtime.types import ProviderResponse, RuntimeConfig

from tests.test_interactive_cli_runtime import _shell


class _FakeProvider:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list = []

    def complete(self, request):
        self.calls.append(request)
        return ProviderResponse(text=self.handler(request), provider="fake", model=request.model or "m")


def _router(handler) -> ProviderRouter:
    config = RuntimeConfig(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m", safety_reviewer_model="m")
    return ProviderRouter({"m": _FakeProvider(handler)}, config)


def _default_handler(req):
    content = req.messages[-1]["content"] if req.messages else ""
    if "sub-queries" in content or '"subqueries"' in content:
        return '{"subqueries": ["vitiligo GWAS loci", "vitiligo autoimmune mechanism"]}'
    return "# Vitiligo Research Brief\n\nKey finding from the literature [1].\n"


def _fake_retriever(query, max_sources):
    slug = query.replace(" ", "_")
    return {"sources": [{"title": f"Paper: {query}", "url": f"http://x/{slug}", "doi": f"10.1/{slug}"}]}


def test_research_produces_cited_report_with_sources():
    router = _router(_default_handler)
    calls = []
    researcher = RuntimeResearcher(
        router,
        retriever=lambda q, n: (calls.append(q) or _fake_retriever(q, n)),
        verifier=lambda text: {"checks": 1, "ok": True},
        num_subqueries=3,
    )
    result = researcher.research("compare two vitiligo phenotype groups via WGS")
    assert result["subqueries"] == ["vitiligo GWAS loci", "vitiligo autoimmune mechanism"]
    assert "Research Brief" in result["report"]
    assert len(result["sources"]) == 2  # one source per sub-query, deduped
    assert len(calls) == 2  # retriever invoked per sub-query (parallel)
    assert result["verification"] == {"checks": 1, "ok": True}


def test_research_without_retriever_still_synthesizes():
    router = _router(_default_handler)
    researcher = RuntimeResearcher(router, retriever=None, verifier=lambda t: {})
    result = researcher.research("any question")
    assert result["sources"] == []
    assert "Research Brief" in result["report"]


def test_research_empty_question_raises():
    router = _router(_default_handler)
    researcher = RuntimeResearcher(router, verifier=lambda t: {})
    with pytest.raises(CouncilError):
        researcher.research("   ")


def test_research_empty_synthesis_raises():
    def handler(req):
        content = req.messages[-1]["content"] if req.messages else ""
        if "sub-queries" in content or '"subqueries"' in content:
            return '{"subqueries": ["q1"]}'
        return ""  # synthesis yields nothing

    researcher = RuntimeResearcher(_router(handler), retriever=_fake_retriever, verifier=lambda t: {})
    with pytest.raises(CouncilError):
        researcher.research("question")


def test_research_emits_real_stages():
    events = []
    researcher = RuntimeResearcher(_router(_default_handler), retriever=_fake_retriever, verifier=lambda t: {})
    researcher.research("q", emit=lambda stage, *, status="running", message="", metadata=None: events.append((stage, status)))
    stages = {s for s, _ in events}
    assert {"Research setup", "Planning", "Merge", "Validation", "Review"}.issubset(stages)


def test_phantom_citations_are_stripped():
    def handler(req):
        content = req.messages[-1]["content"] if req.messages else ""
        if "sub-queries" in content or '"subqueries"' in content:
            return '{"subqueries": ["q1", "q2"]}'
        return "# Brief\n\nClaim A [1]. Claim B [3]. Claim C [2].\n"  # only 2 sources exist

    researcher = RuntimeResearcher(_router(handler), retriever=_fake_retriever, verifier=lambda t: {})
    result = researcher.research("question with two subqueries")
    assert len(result["sources"]) == 2
    report = result["report"]
    assert "[1]" in report and "[2]" in report  # valid citations kept
    assert "[3]" not in report  # phantom citation removed


def test_no_source_report_labeled_model_only():
    def handler(req):
        content = req.messages[-1]["content"] if req.messages else ""
        if "sub-queries" in content or '"subqueries"' in content:
            return '{"subqueries": ["q1"]}'
        return "# Brief\n\nA claim [1].\n"  # model invents a citation with no sources

    researcher = RuntimeResearcher(_router(handler), retriever=None, verifier=lambda t: {})
    result = researcher.research("question")
    assert result["sources"] == []
    assert "[1]" not in result["report"]  # no sources -> citation stripped
    assert "model knowledge only" in result["report"]


def test_research_command_registered_and_renders(tmp_path):
    shell, output = _shell(tmp_path)

    class _FakeResearcher:
        def research(self, question, **kwargs):
            return {
                "question": question,
                "subqueries": ["a", "b"],
                "sources": [{"title": "T1"}, {"title": "T2"}],
                "report": "# Result\n\nFindings here.",
                "verification": {"ok": True},
            }

    shell._researcher = _FakeResearcher()
    shell.handle_line("/research compare phenotype groups")
    text = output.getvalue()
    assert "Research Brief" in text
    assert "2 source" in text
