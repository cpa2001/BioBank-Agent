"""Test read_pdf skill — PDF text and table extraction."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestReadPdfImport:
    """Test read_pdf skill registration."""

    def test_read_pdf_imports(self):
        """Test that read_pdf skill can be imported."""
        from biobank_agent.skills.read_pdf import read_pdf
        assert callable(read_pdf)
        assert read_pdf._skill_name == "read_pdf"

    def test_read_pdf_schema(self):
        """Test read_pdf has correct parameters."""
        from biobank_agent.skills.read_pdf import read_pdf
        schema = read_pdf._skill_schema
        func_def = schema["function"]
        assert "path" in func_def["parameters"]["properties"]
        assert "max_pages" in func_def["parameters"]["properties"]
        assert "path" in func_def["parameters"]["required"]


class TestReadPdfFileErrors:
    """Test file validation errors."""

    def test_file_not_found(self, tmp_path):
        """Non-existent path -> error."""
        from biobank_agent.skills.read_pdf import read_pdf

        result = read_pdf(path=str(tmp_path / "does_not_exist.pdf"), ctx=None)

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_not_a_pdf_file(self, tmp_path):
        """File that is not a .pdf -> error."""
        from biobank_agent.skills.read_pdf import read_pdf

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello")

        result = read_pdf(path=str(txt_file), ctx=None)

        assert "error" in result
        assert "Not a PDF" in result["error"]


class TestReadPdfExtraction:
    """Test PDF text and metadata extraction with mocked pymupdf."""

    def _mock_page(self, text="Page content here.", tables=None):
        """Create a mock pymupdf page."""
        page = MagicMock()
        page.get_text.return_value = text

        # Mock table finder
        if tables:
            tab_finder = MagicMock()
            mock_tables = []
            for t in tables:
                mt = MagicMock()
                mt.extract.return_value = t
                mock_tables.append(mt)
            tab_finder.tables = mock_tables
            page.find_tables.return_value = tab_finder
        else:
            tab_finder = MagicMock()
            tab_finder.tables = []
            page.find_tables.return_value = tab_finder

        return page

    def _mock_doc(self, pages, metadata=None):
        """Create a mock pymupdf document."""
        doc = MagicMock()
        doc.__len__ = lambda self: len(pages)
        doc.__getitem__ = lambda self, idx: pages[idx]
        doc.is_encrypted = False
        doc.metadata = metadata or {}
        return doc

    @patch("pymupdf.open")
    def test_single_page_extraction(self, mock_open, tmp_path):
        """Single-page PDF -> text extracted."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        page = self._mock_page(text="Introduction\nThis paper studies biomarkers.")
        doc = self._mock_doc([page])
        mock_open.return_value = doc

        result = read_pdf(path=str(pdf_file), ctx=None)

        assert "error" not in result
        assert result["n_pages"] == 1
        assert result["pages_read"] == 1
        assert "biomarkers" in result["text"]

    @patch("pymupdf.open")
    def test_multi_page_with_max_pages(self, mock_open, tmp_path):
        """max_pages limits how many pages are read."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "big.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        pages = [self._mock_page(text=f"Page {i+1} content") for i in range(10)]
        doc = self._mock_doc(pages)
        mock_open.return_value = doc

        result = read_pdf(path=str(pdf_file), max_pages=3, ctx=None)

        assert result["n_pages"] == 10
        assert result["pages_read"] == 3
        assert "Page 1" in result["text"]
        assert "Page 3" in result["text"]
        assert "Page 4" not in result["text"]

    @patch("pymupdf.open")
    def test_max_pages_zero_reads_all(self, mock_open, tmp_path):
        """max_pages=0 -> read all pages."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "full.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        pages = [self._mock_page(text=f"Page {i+1}") for i in range(5)]
        doc = self._mock_doc(pages)
        mock_open.return_value = doc

        result = read_pdf(path=str(pdf_file), max_pages=0, ctx=None)

        assert result["pages_read"] == 5

    @patch("pymupdf.open")
    def test_table_extraction(self, mock_open, tmp_path):
        """Tables on a page are extracted as structured data."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "tables.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        table_data = [
            ["Biomarker", "P-value"],
            ["HbA1c", "1.2e-10"],
            ["BMI", "3.4e-8"],
        ]
        page = self._mock_page(text="Results", tables=[table_data])
        doc = self._mock_doc([page])
        mock_open.return_value = doc

        result = read_pdf(path=str(pdf_file), ctx=None)

        assert len(result["tables"]) == 1
        assert result["tables"][0]["page"] == 1
        assert result["tables"][0]["rows"][0] == ["Biomarker", "P-value"]

    @patch("pymupdf.open")
    def test_metadata_extraction(self, mock_open, tmp_path):
        """PDF metadata (title, author) is extracted."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "meta.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        page = self._mock_page(text="Content")
        meta = {
            "title": "Biomarker Discovery in UK Biobank",
            "author": "Smith et al.",
            "subject": "",
            "keywords": "biomarkers",
            "creator": "LaTeX",
            "producer": "pdfTeX",
        }
        doc = self._mock_doc([page], metadata=meta)
        mock_open.return_value = doc

        result = read_pdf(path=str(pdf_file), ctx=None)

        assert result["metadata"]["title"] == "Biomarker Discovery in UK Biobank"
        assert result["metadata"]["author"] == "Smith et al."
        assert "subject" not in result["metadata"]  # empty string filtered

    @patch("pymupdf.open")
    def test_encrypted_pdf_error(self, mock_open, tmp_path):
        """Encrypted PDF that can't be authenticated -> error."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "encrypted.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        doc = MagicMock()
        doc.is_encrypted = True
        doc.authenticate.return_value = False
        mock_open.return_value = doc

        result = read_pdf(path=str(pdf_file), ctx=None)

        assert "error" in result
        assert "encrypted" in result["error"].lower()

    @patch("pymupdf.open")
    def test_open_failure(self, mock_open, tmp_path):
        """pymupdf.open raises exception -> error."""
        from biobank_agent.skills.read_pdf import read_pdf

        pdf_file = tmp_path / "corrupt.pdf"
        pdf_file.write_bytes(b"%PDF-fake")

        mock_open.side_effect = RuntimeError("Corrupt file")

        result = read_pdf(path=str(pdf_file), ctx=None)

        assert "error" in result
        assert "Cannot open PDF" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
