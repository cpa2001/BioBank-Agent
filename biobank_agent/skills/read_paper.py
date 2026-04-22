"""Read paper skill — structured critical analysis of a scientific paper.

Extracts text from a PDF or fetches by DOI, then produces an analysis
template covering summary, methods, evidence quality, and UKB relevance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)

# Common ICD-10 chapter prefixes for detection
_ICD10_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")

# Common UK Biobank field-like references (e.g. "field 30750", "UKB field ID 21001")
_UKB_FIELD_RE = re.compile(r"(?:field|UKB|UK\s*Biobank)\s*(?:field\s*)?(?:ID\s*)?(\d{3,6})", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Analysis template builder
# ---------------------------------------------------------------------------

_FOCUS_PROMPTS: dict[str, str] = {
    "general": (
        "Provide a comprehensive analysis covering study design, statistical methods, "
        "main findings, limitations, and relevance to UK Biobank research."
    ),
    "methods": (
        "Focus deeply on the methodology: study design, inclusion/exclusion criteria, "
        "statistical models, multiple-testing corrections, sensitivity analyses, and "
        "potential methodological weaknesses."
    ),
    "biomarkers": (
        "Focus on biomarkers studied: which biomarkers were measured, assay platforms, "
        "reference ranges, associations found, and which UK Biobank biomarker fields overlap."
    ),
    "ukb-relevance": (
        "Focus on UK Biobank relevance: does this paper use UKB data? What UKB fields "
        "and ICD-10 codes are referenced? How could their approach be replicated or "
        "extended using our UKB access?"
    ),
    "genetics": (
        "Focus on genetic methods and findings: GWAS design, SNPs identified, heritability "
        "estimates, polygenic risk scores, Mendelian randomisation, and genetic overlap "
        "with other traits."
    ),
    "epidemiology": (
        "Focus on epidemiological design: cohort definition, exposure/outcome measurement, "
        "confounding adjustment, selection bias, and generalisability to the UK Biobank population."
    ),
}


def _build_analysis_template(focus: str) -> str:
    """Return a structured analysis template for the LLM to fill in."""
    focus_instruction = _FOCUS_PROMPTS.get(focus, _FOCUS_PROMPTS["general"])

    return f"""## Structured Paper Analysis

Please analyse this paper using the following framework.
**Focus**: {focus_instruction}

### 1. Paper Summary
- **Title**:
- **Authors / Year / Venue**:
- **Key Finding (one sentence)**:
- **Study Type** (RCT / prospective cohort / cross-sectional / Mendelian randomisation / meta-analysis / other):

### 2. Methods Evaluation
- **Study Design**:
- **Population / Sample Size**:
- **Primary Outcome**:
- **Statistical Approach**:
- **Multiple Testing Correction**:
- **Strengths**:
- **Weaknesses**:

### 3. Claims vs Evidence
- **Main Claims**:
- **Supporting Evidence**:
- **Effect Sizes (with 95% CI)**:
- **Are conclusions proportionate to the evidence?** (yes/no + explanation):

### 4. UK Biobank Relevance
- **Uses UKB Data?** (yes/no):
- **ICD-10 Codes Referenced**:
- **UKB Fields Referenced**:
- **Biomarkers / Phenotypes That Overlap with UKB**:
- **Could this analysis be replicated with our UKB data?** (yes/no + what's needed):

### 5. Key Takeaways for Our Research
- **Actionable Insights**:
- **Suggested Follow-up Analyses**:
- **Caveats / Red Flags**:
"""


def _extract_icd10_codes(text: str) -> list[str]:
    """Extract unique ICD-10 codes mentioned in the text."""
    codes = _ICD10_RE.findall(text)
    # Filter out things that are clearly not ICD-10 (e.g. P values like P001)
    icd_codes = []
    seen = set()
    for code in codes:
        # Valid ICD-10 chapters: A-Z (roughly A00-Z99)
        chapter = code[0]
        if chapter in "XYZWVU" and not code.startswith(("V", "W", "X", "Y", "Z")):
            continue
        if code not in seen:
            icd_codes.append(code)
            seen.add(code)
    return icd_codes[:50]  # cap at 50


def _extract_ukb_field_refs(text: str) -> list[str]:
    """Extract UK Biobank field ID references from text."""
    matches = _UKB_FIELD_RE.findall(text)
    return list(dict.fromkeys(matches))[:30]  # unique, capped


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="read_paper",
    description="Deep critical analysis of a scientific paper. Evaluates claims, methods, "
                "evidence quality, and relevance to UK Biobank research.",
    parameters={
        "paper_path_or_doi": {
            "type": "string",
            "description": "Path to PDF or DOI to analyze",
        },
        "focus": {
            "type": "string",
            "description": (
                "Specific aspect to focus on: 'general', 'methods', 'biomarkers', "
                "'ukb-relevance', 'genetics', 'epidemiology'"
            ),
            "default": "general",
        },
    },
    required=["paper_path_or_doi"],
)
def read_paper(paper_path_or_doi: str, focus: str = "general", *, ctx=None) -> dict:
    """Analyse a scientific paper and return a structured evaluation.

    If a DOI is provided, the paper is fetched first via :func:`fetch_paper`.
    If a file path is provided, text is extracted via :func:`read_pdf`.

    The function returns the paper text (truncated) and an analysis template
    that the agent LLM should complete.

    Returns
    -------
    dict
        ``{"summary": {...}, "paper_text": str, "analysis_template": str,
           "icd10_codes_mentioned": [...], "ukb_fields_mentioned": [...]}``
    """
    paper_path_or_doi = paper_path_or_doi.strip()
    paper_text = ""
    summary: dict[str, Any] = {}
    pdf_path: str | None = None

    # ------------------------------------------------------------------
    # 1. Obtain paper text
    # ------------------------------------------------------------------

    is_doi = re.match(r"^10\.\d{4,9}/", paper_path_or_doi)
    is_url = paper_path_or_doi.startswith("http")

    if is_doi or is_url:
        # Fetch via fetch_paper
        try:
            from biobank_agent.skills.fetch_paper import fetch_paper

            paper_data = fetch_paper(identifier=paper_path_or_doi, ctx=ctx)
            paper_text = paper_data.get("full_text", "") or paper_data.get("abstract", "")
            summary = {
                "title": paper_data.get("title", ""),
                "authors": paper_data.get("authors", []),
                "doi": paper_data.get("doi", ""),
                "pdf_path": paper_data.get("pdf_path"),
            }
            pdf_path = paper_data.get("pdf_path")
            if paper_data.get("warning"):
                summary["access_warning"] = paper_data["warning"]
        except Exception as exc:
            return {"error": f"Failed to fetch paper: {exc}"}
    else:
        # Treat as file path
        try:
            from biobank_agent.skills.read_pdf import read_pdf

            pdf_result = read_pdf(path=paper_path_or_doi, max_pages=0, ctx=ctx)
            if "error" in pdf_result:
                return {"error": pdf_result["error"]}
            paper_text = pdf_result.get("text", "")
            pdf_path = paper_path_or_doi
            meta = pdf_result.get("metadata", {})
            summary = {
                "title": meta.get("title", ""),
                "authors": [meta.get("author", "")] if meta.get("author") else [],
                "n_pages": pdf_result.get("n_pages", 0),
                "pdf_path": paper_path_or_doi,
            }
        except Exception as exc:
            return {"error": f"Failed to read PDF: {exc}"}

    if not paper_text:
        return {
            "error": "No text could be extracted from the paper.",
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # 2. Extract structured references from the text
    # ------------------------------------------------------------------

    icd10_codes = _extract_icd10_codes(paper_text)
    ukb_fields = _extract_ukb_field_refs(paper_text)

    # ------------------------------------------------------------------
    # 3. Build analysis template for the agent LLM
    # ------------------------------------------------------------------

    analysis_template = _build_analysis_template(focus)

    # Truncate paper text to a reasonable size for LLM context
    max_text_chars = 8000
    text_truncated = len(paper_text) > max_text_chars
    display_text = paper_text[:max_text_chars]
    if text_truncated:
        display_text += "\n\n[... paper text truncated for analysis ...]"

    return {
        "summary": summary,
        "paper_text": display_text,
        "full_text_length": len(paper_text),
        "text_truncated": text_truncated,
        "analysis_template": analysis_template,
        "icd10_codes_mentioned": icd10_codes,
        "ukb_fields_mentioned": ukb_fields,
        "focus": focus,
    }
