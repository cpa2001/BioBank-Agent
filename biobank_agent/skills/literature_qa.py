"""Literature QA skill — citation-first scientific search with provenance.

Wraps PaperQA2 (github.com/Future-House/paper-qa) as an optional biobank
agent skill. Provides indexed search over local PDF corpus with full
citation provenance (DOI, page, passage, confidence).

Source paper:
    Skarlinski et al., "Language Agents Achieve Superhuman Synthesis of
    Scientific Knowledge" (2024). github.com/Future-House/paper-qa

Design reference:
    Claude research doc §C3: "PaperQA2... Beats PhD/postdoc-level human
    accuracy on LitQA2; SOTA on RAG-QA Arena Science."

Graceful fallback:
    If paper-qa is not installed, returns a structured response with
    install instructions rather than raising an ImportError.

Integration:
    Citations returned by this skill feed into the Action Graph
    for claim-level provenance tracking.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)

# Default corpus location (docs/related_works/ contains project PDFs)
_DEFAULT_CORPUS_DIR = Path(__file__).parent.parent.parent / "docs" / "related_works"


# ── Availability check ─────────────────────────────────────────────────────

try:
    from paperqa import Docs, Settings
    _PAPERQA_AVAILABLE = True
except ImportError:
    _PAPERQA_AVAILABLE = False


def is_paperqa_available() -> bool:
    """Check if paper-qa is installed."""
    return _PAPERQA_AVAILABLE


# ── Internal helpers ─────────────────────────────────────────────────────────


def _build_docs_index(corpus_dir: Path, settings: Optional[Any] = None) -> Any:
    """Build or load a PaperQA2 document index from a directory of PDFs.

    Args:
        corpus_dir: Directory containing PDF files to index
        settings: Optional PaperQA2 Settings object

    Returns:
        A Docs instance with indexed papers
    """
    if not _PAPERQA_AVAILABLE:
        return None

    docs = Docs()

    # Index all PDFs in the corpus directory
    pdf_files = list(corpus_dir.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", corpus_dir)
        return docs

    for pdf_path in pdf_files:
        try:
            docs.add(str(pdf_path))
            logger.debug("Indexed: %s", pdf_path.name)
        except Exception as e:
            logger.warning("Failed to index %s: %s", pdf_path.name, e)

    return docs


def _compute_provenance_hash(query: str, answer: str, sources: list[dict]) -> str:
    """Compute SHA-256 provenance hash for a literature QA result."""
    content = f"{query}|{answer}|{len(sources)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Skill definition ─────────────────────────────────────────────────────────


@skill(
    name="literature_qa",
    description=(
        "Search scientific literature with full citation provenance. "
        "Uses PaperQA2 to answer research questions from indexed papers "
        "with DOI, page, and passage-level attribution."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Research question to answer from the literature",
            "required": True,
        },
        "max_sources": {
            "type": "integer",
            "description": "Maximum number of papers to cite in the answer",
            "default": 5,
        },
        "corpus_dir": {
            "type": "string",
            "description": "Path to directory containing PDFs to search (default: docs/related_works/)",
            "default": None,
        },
    },
)
def literature_qa(
    query: str,
    max_sources: int = 5,
    corpus_dir: Optional[str] = None,
    *,
    ctx: Any = None,
) -> dict[str, Any]:
    """Execute literature search with citation provenance tracking.

    Args:
        query: Research question to answer from the literature
        max_sources: Maximum number of papers to cite
        corpus_dir: Optional path to PDF directory
        ctx: Skill execution context (injected by registry)

    Returns:
        Dictionary with:
            - answer: Synthesized answer from papers
            - citations: List of {doi, title, page, passage, confidence}
            - provenance_hash: SHA-256 hash for evidence lattice
            - n_papers_indexed: Number of papers in the index
            - status: "success" or "error"

    If paper-qa is not installed, returns error dict with install instructions.
    """
    # ── Graceful fallback ──
    if not _PAPERQA_AVAILABLE:
        return {
            "status": "unavailable",
            "error": "paper-qa is not installed",
            "install": "pip install 'biobank-agent[literature]'",
            "fallback_suggestion": (
                "Use the 'fetch_paper' skill to download individual papers, "
                "or 'web_search' for online literature search."
            ),
        }

    # ── Resolve corpus directory ──
    if corpus_dir:
        papers_dir = Path(corpus_dir)
    else:
        papers_dir = _DEFAULT_CORPUS_DIR

    if not papers_dir.exists():
        return {
            "status": "error",
            "error": f"Corpus directory not found: {papers_dir}",
            "suggestion": "Provide a valid corpus_dir or add PDFs to docs/related_works/",
        }

    # ── Build index and query ──
    try:
        docs = _build_docs_index(papers_dir)
        if docs is None:
            return {
                "status": "error",
                "error": "Failed to build document index",
            }

        # Execute the query
        answer_response = docs.query(
            query,
            k=max_sources,
        )

        # Extract structured citations
        citations = []
        for context in getattr(answer_response, "contexts", []):
            doc = getattr(context, "doc", None)
            citation = {
                "title": (doc.get("title", "Unknown") if isinstance(doc, dict)
                          else getattr(doc, "title", "Unknown")),
                "doi": (doc.get("doi", "") if isinstance(doc, dict)
                        else getattr(doc, "doi", "")),
                "page": getattr(context, "page", None),
                "passage": getattr(context, "text", "")[:500],  # Truncate long passages
                "confidence": getattr(context, "score", 0.0),
            }
            citations.append(citation)

        # Compute provenance hash
        answer_text = getattr(answer_response, "answer", str(answer_response))
        provenance_hash = _compute_provenance_hash(query, answer_text, citations)

        return {
            "status": "success",
            "answer": answer_text,
            "citations": citations[:max_sources],
            "provenance_hash": provenance_hash,
            "n_papers_indexed": len(list(papers_dir.glob("**/*.pdf"))),
            "confidence": getattr(answer_response, "confidence", None),
            "query": query,
        }

    except Exception as e:
        logger.error("Literature QA failed: %s", e, exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "suggestion": "Check that PDFs in the corpus are readable and not corrupted.",
        }
