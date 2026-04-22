"""Read PDF skill — extract text and tables from PDF documents.

Uses PyMuPDF (``pymupdf`` / ``fitz``) for fast, high-quality extraction.
Works with scientific papers, UK Biobank documentation, and general PDFs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tables_from_page(page: Any) -> list[list[list[str]]]:
    """Try to extract tables from a single page using PyMuPDF's table finder.

    Returns a list of tables, where each table is a list of rows, and each
    row is a list of cell strings.
    """
    tables: list[list[list[str]]] = []
    try:
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            extracted = table.extract()
            # Clean up None cells
            clean_rows = []
            for row in extracted:
                clean_rows.append([str(cell) if cell is not None else "" for cell in row])
            tables.append(clean_rows)
    except Exception:
        # find_tables() may not be available in older pymupdf versions
        pass
    return tables


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="read_pdf",
    description="Extract text and tables from a PDF file. Works with scientific papers, "
                "UK Biobank documentation, and any PDF document.",
    parameters={
        "path": {"type": "string", "description": "Path to the PDF file"},
        "max_pages": {
            "type": "integer",
            "description": "Maximum pages to read (0 = all)",
            "default": 0,
        },
    },
    required=["path"],
)
def read_pdf(path: str, max_pages: int = 0, *, ctx=None) -> dict:
    """Extract text and tables from a PDF file.

    Parameters
    ----------
    path : str
        Filesystem path to the PDF.
    max_pages : int
        If > 0, only the first *max_pages* pages are read.
    ctx : optional
        Skill context (unused beyond convention).

    Returns
    -------
    dict
        ``{"path": ..., "n_pages": int, "pages_read": int, "text": str,
           "tables": [...], "metadata": {...}}``
    """
    pdf_path = Path(path).expanduser().resolve()

    if not pdf_path.exists():
        return {"error": f"File not found: {pdf_path}"}
    if not pdf_path.suffix.lower() == ".pdf":
        return {"error": f"Not a PDF file: {pdf_path}"}

    try:
        import pymupdf  # PyMuPDF >= 1.24 exposes this namespace
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF versions
        except ImportError:
            return {"error": "pymupdf (PyMuPDF) is not installed. Run: pip install pymupdf"}

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        return {"error": f"Cannot open PDF: {exc}"}

    # Check for encryption
    if doc.is_encrypted:
        try:
            if not doc.authenticate(""):
                return {"error": "PDF is encrypted and requires a password"}
        except Exception:
            return {"error": "PDF is encrypted and requires a password"}

    n_pages = len(doc)
    pages_to_read = n_pages if max_pages <= 0 else min(max_pages, n_pages)

    text_parts: list[str] = []
    all_tables: list[dict] = []

    for page_idx in range(pages_to_read):
        page = doc[page_idx]

        # --- Text ---
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(f"--- Page {page_idx + 1} ---\n{page_text.strip()}")

        # --- Tables ---
        page_tables = _extract_tables_from_page(page)
        for t_idx, table_data in enumerate(page_tables):
            all_tables.append({
                "page": page_idx + 1,
                "table_index": t_idx,
                "rows": table_data,
            })

    # --- Metadata ---
    metadata: dict[str, Any] = {}
    try:
        raw_meta = doc.metadata
        if raw_meta:
            for key in ("title", "author", "subject", "keywords", "creator", "producer"):
                val = raw_meta.get(key, "")
                if val:
                    metadata[key] = val
    except Exception:
        pass

    doc.close()

    full_text = "\n\n".join(text_parts)

    return {
        "path": str(pdf_path),
        "n_pages": n_pages,
        "pages_read": pages_to_read,
        "text": full_text,
        "tables": all_tables,
        "metadata": metadata,
    }
