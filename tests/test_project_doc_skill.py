"""Tests for curated project documentation access."""

from pathlib import Path

from biobank_agent.skills.project_doc import _title_for, project_doc


def test_project_doc_lists_curated_markdown():
    result = project_doc(mode="list", limit=50)
    paths = {entry["path"] for entry in result["results"]}

    assert result["status"] == "success"
    assert "README.md" in paths
    assert "docs/data/UKB_DATA_REFERENCE.md" in paths
    assert all("deep_research" not in path for path in paths)


def test_project_doc_searches_document_content():
    result = project_doc(mode="search", query="plugin integration", limit=5)

    assert result["status"] == "success"
    assert result["results"]
    assert any("PLUGIN_INTEGRATION" in item["path"] or item["path"] == "README.md" for item in result["results"])


def test_project_doc_reads_allowed_file_with_truncation():
    result = project_doc(mode="read", path="docs/data/UKB_DATA_REFERENCE.md", max_chars=600)

    assert result["status"] == "success"
    assert result["path"] == "docs/data/UKB_DATA_REFERENCE.md"
    assert "UK Biobank" in result["content"]
    assert len(result["content"]) <= 600


def test_project_doc_rejects_uncurated_paths():
    outside = project_doc(mode="read", path="../.env")
    ignored = project_doc(mode="read", path="docs/deep_research/GPT-deep-research-report.md")

    assert outside["status"] == "error"
    assert ignored["status"] == "error"


def test_project_doc_infers_modes():
    listed = project_doc(limit=3)
    searched = project_doc(query="data reference", limit=3)
    read = project_doc(path="README.md", max_chars=1000)

    assert listed["mode"] == "list"
    assert searched["mode"] == "search"
    assert read["mode"] == "read"


def test_project_doc_title_sanitizes_sensitive_headings():
    """Redaction of titles must be STRUCTURAL, not accidental: a document whose
    heading contains a credential term must never leak it through the title
    field surfaced by list/search/read. Tested at the source (_title_for)."""
    title = _title_for(Path("fake.md"), "# Setup API_KEY rotation\n\nbody text\n")

    assert "api_key" not in title.lower()
    assert "api key" not in title.lower()
    assert "credential" in title.lower()  # the redacted replacement is present

    # Heading-less docs fall back to the (sanitized) filename stem.
    stem_title = _title_for(Path("secret_api_key.md"), "no headings here\n")
    assert "api_key" not in stem_title.lower()


def test_project_doc_redacts_sensitive_config_terms_from_transcript():
    read = project_doc(mode="read", path="README.md", max_chars=50000)
    search = project_doc(mode="search", query="api key", limit=5)

    assert "api_key" not in read["content"].lower()
    assert "api key" not in read["content"].lower()
    assert "api_key" not in str(search).lower()
    assert "api key" not in str(search).lower()
