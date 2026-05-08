"""Tests for PaperQA2 literature skill (Phase 5).

Tests graceful fallback when paper-qa is not installed,
mock result parsing, and provenance hash computation.
"""

import importlib
import sys

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from types import SimpleNamespace
import biobank_agent.skills.literature_qa as lit_mod
from biobank_agent.skills.literature_qa import (
    literature_qa,
    is_paperqa_available,
    citations_to_evidence_nodes,
    _compute_provenance_hash,
)


class TestPaperQAAvailability:
    """Test availability detection."""

    def test_availability_is_bool(self):
        """Should return boolean."""
        result = is_paperqa_available()
        assert isinstance(result, bool)

    def test_import_with_fake_paperqa_marks_available(self):
        sentinel = object()
        original_paperqa = sys.modules.get("paperqa", sentinel)
        sys.modules["paperqa"] = SimpleNamespace(Docs=object, Settings=object)
        try:
            reloaded = importlib.reload(lit_mod)
            assert reloaded._PAPERQA_AVAILABLE is True
        finally:
            if original_paperqa is sentinel:
                sys.modules.pop("paperqa", None)
            else:
                sys.modules["paperqa"] = original_paperqa
            importlib.reload(lit_mod)


class TestGracefulFallback:
    """Test behavior when paper-qa is NOT installed."""

    def test_returns_unavailable_status(self):
        """Should return install instructions, not raise."""
        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", False):
            result = literature_qa(query="What causes diabetes?")
            assert result["status"] == "unavailable"
            assert "paper-qa" in result["error"]
            assert "pip install" in result["install"]
            assert "fallback_suggestion" in result

    def test_fallback_has_alternative_skills(self):
        """Fallback should suggest alternative skills."""
        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", False):
            result = literature_qa(query="test")
            assert "fetch_paper" in result["fallback_suggestion"] or "web_search" in result["fallback_suggestion"]


class TestCorpusDirectory:
    """Test corpus directory handling."""

    def test_missing_corpus_dir_returns_error(self, tmp_path):
        """Non-existent corpus dir should return error."""
        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", True):
            with patch("biobank_agent.skills.literature_qa._build_docs_index", return_value=None):
                result = literature_qa(
                    query="test",
                    corpus_dir=str(tmp_path / "nonexistent")
                )
                assert result["status"] == "error"
                assert "not found" in result["error"]

    def test_valid_corpus_index_build_failure_returns_error(self, tmp_path):
        corpus = tmp_path / "papers"
        corpus.mkdir()

        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", True):
            with patch("biobank_agent.skills.literature_qa._build_docs_index", return_value=None):
                result = literature_qa(query="test", corpus_dir=str(corpus))

        assert result == {"status": "error", "error": "Failed to build document index"}

    def test_custom_corpus_dir(self, tmp_path):
        """Should accept custom corpus directory."""
        # Create a valid but empty directory
        corpus = tmp_path / "papers"
        corpus.mkdir()
        # Paper QA available but will fail due to no PDFs
        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", True):
            with patch("biobank_agent.skills.literature_qa._build_docs_index") as mock_build:
                mock_docs = MagicMock()
                mock_docs.query.return_value = MagicMock(
                    answer="No papers found",
                    contexts=[],
                    confidence=0.0,
                )
                mock_build.return_value = mock_docs
                result = literature_qa(query="test", corpus_dir=str(corpus))
                # Should attempt to build index for custom dir
                mock_build.assert_called_once()


class TestDocsIndexBuilder:
    """Test PaperQA index construction without requiring real PaperQA."""

    def test_unavailable_empty_and_pdf_add_failure_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lit_mod, "_PAPERQA_AVAILABLE", False)
        assert lit_mod._build_docs_index(tmp_path) is None

        class FakeDocs:
            def __init__(self):
                self.added = []

            def add(self, path):
                if path.endswith("bad.pdf"):
                    raise RuntimeError("bad pdf")
                self.added.append(Path(path).name)

        monkeypatch.setattr(lit_mod, "_PAPERQA_AVAILABLE", True)
        monkeypatch.setattr(lit_mod, "Docs", FakeDocs, raising=False)

        empty_docs = lit_mod._build_docs_index(tmp_path)
        assert empty_docs.added == []

        (tmp_path / "ok.pdf").write_text("ok", encoding="utf-8")
        (tmp_path / "bad.pdf").write_text("bad", encoding="utf-8")
        indexed = lit_mod._build_docs_index(tmp_path)
        assert indexed.added == ["ok.pdf"]


class TestProvenanceHash:
    """Test provenance hash computation."""

    def test_hash_is_hex(self):
        """Hash should be 16-char hex string."""
        h = _compute_provenance_hash("query", "answer", [{"doi": "test"}])
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_queries_different_hashes(self):
        """Different queries should produce different hashes."""
        h1 = _compute_provenance_hash("query1", "answer", [])
        h2 = _compute_provenance_hash("query2", "answer", [])
        assert h1 != h2

    def test_same_input_same_hash(self):
        """Same inputs should produce same hash."""
        h1 = _compute_provenance_hash("q", "a", [{"x": 1}])
        h2 = _compute_provenance_hash("q", "a", [{"x": 1}])
        assert h1 == h2


class TestCitationsToEvidenceNodes:
    """Test conversion of citations to evidence lattice format."""

    def test_successful_result_converts(self):
        """Successful result should produce evidence nodes."""
        result = {
            "status": "success",
            "citations": [
                {"doi": "10.1038/test", "title": "Test Paper", "passage": "Key finding", "confidence": 0.8},
                {"doi": "10.1234/other", "title": "Other", "passage": "Another finding", "confidence": 0.6},
            ],
        }
        nodes = citations_to_evidence_nodes(result, claim_text="Test claim")
        assert len(nodes) == 2
        assert nodes[0]["evidence_type"] == "paper"
        assert nodes[0]["confidence"] == 0.8
        assert nodes[0]["direction"] == "supports"
        assert len(nodes[0]["source_hash"]) == 16

    def test_failed_result_returns_empty(self):
        """Failed result should return empty list."""
        result = {"status": "error", "error": "something failed"}
        nodes = citations_to_evidence_nodes(result)
        assert nodes == []

    def test_unavailable_result_returns_empty(self):
        """Unavailable status should return empty list."""
        result = {"status": "unavailable"}
        nodes = citations_to_evidence_nodes(result)
        assert nodes == []

    def test_empty_citations_returns_empty(self):
        """Success with no citations should return empty."""
        result = {"status": "success", "citations": []}
        nodes = citations_to_evidence_nodes(result)
        assert nodes == []

    def test_missing_doi_uses_title_and_default_confidence(self):
        result = {
            "status": "success",
            "citations": [{"title": "Title Only", "passage": "passage"}],
        }

        nodes = citations_to_evidence_nodes(result)

        assert nodes[0]["source_ref"] == "Title Only"
        assert nodes[0]["confidence"] == 0.5


class TestMockPaperQAExecution:
    """Test with mocked PaperQA2 execution."""

    def test_successful_query(self, tmp_path):
        """Mocked successful query should return structured result."""
        corpus = tmp_path / "papers"
        corpus.mkdir()
        (corpus / "test.pdf").write_text("fake pdf")  # Just needs to exist

        mock_response = MagicMock()
        mock_response.answer = "HbA1c is a strong predictor of T2D"
        mock_response.confidence = 0.85
        mock_context = MagicMock()
        mock_context.doc = {"title": "Diabetes Review", "doi": "10.1038/test"}
        mock_context.page = 5
        mock_context.text = "HbA1c levels above 48 indicate diabetes"
        mock_context.score = 0.9
        mock_response.contexts = [mock_context]

        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", True):
            with patch("biobank_agent.skills.literature_qa._build_docs_index") as mock_build:
                mock_docs = MagicMock()
                mock_docs.query.return_value = mock_response
                mock_build.return_value = mock_docs

                result = literature_qa(
                    query="What predicts type 2 diabetes?",
                    corpus_dir=str(corpus),
                )

                assert result["status"] == "success"
                assert "HbA1c" in result["answer"]
                assert len(result["citations"]) == 1
                assert result["citations"][0]["doi"] == "10.1038/test"
                assert len(result["provenance_hash"]) == 16

    def test_default_corpus_object_doc_and_string_answer(self, tmp_path):
        """Default corpus path and object-style docs should parse cleanly."""
        corpus = tmp_path / "papers"
        corpus.mkdir()
        (corpus / "paper.pdf").write_text("fake pdf", encoding="utf-8")

        class Response:
            confidence = None
            contexts = [
                SimpleNamespace(
                    doc=SimpleNamespace(title="Object Paper", doi="10.1000/object"),
                    page=7,
                    text="x" * 600,
                    score=0.77,
                )
            ]

            def __str__(self):
                return "stringified answer"

        mock_docs = MagicMock()
        mock_docs.query.return_value = Response()

        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", True):
            with patch("biobank_agent.skills.literature_qa._DEFAULT_CORPUS_DIR", corpus):
                with patch("biobank_agent.skills.literature_qa._build_docs_index", return_value=mock_docs):
                    result = literature_qa(query="What is known?", max_sources=1)

        assert result["status"] == "success"
        assert result["answer"] == "stringified answer"
        assert result["citations"][0]["title"] == "Object Paper"
        assert result["citations"][0]["doi"] == "10.1000/object"
        assert len(result["citations"][0]["passage"]) == 500
        assert result["n_papers_indexed"] == 1

    def test_query_exception_returns_structured_error(self, tmp_path):
        corpus = tmp_path / "papers"
        corpus.mkdir()

        mock_docs = MagicMock()
        mock_docs.query.side_effect = RuntimeError("query failed")

        with patch("biobank_agent.skills.literature_qa._PAPERQA_AVAILABLE", True):
            with patch("biobank_agent.skills.literature_qa._build_docs_index", return_value=mock_docs):
                result = literature_qa(query="bad question", corpus_dir=str(corpus))

        assert result["status"] == "error"
        assert result["error"] == "query failed"
        assert result["query"] == "bad question"
