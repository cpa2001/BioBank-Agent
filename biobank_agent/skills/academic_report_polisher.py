"""Academic report polishing skill and report-generation hook helpers."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill


FORBIDDEN_MAIN_BODY_PATTERNS = (
    "Recorded quantitative outputs",
    "np.float64",
    "np.int64",
    "not recorded cases; not recorded controls",
    "Predictive model | not recorded",
    "analysis completed -- see details below",
    "top finding: .",
    "cohort cohort",
    "visualization visualization",
    "--------=-------",
)

AI_LIKE_REPLACEMENTS = {
    "rich resource": "dataset",
    "leverage": "use",
    "actionable insights": "interpretable results",
    "notable findings": "main findings",
    "deployment-ready": "ready for review",
    "these findings support": "these results are consistent with",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _split_main_appendix(text: str) -> tuple[str, str]:
    match = re.search(r"(?im)^##\s+Execution Appendix\b", text)
    if not match:
        return text, ""
    return text[: match.start()].rstrip() + "\n", text[match.start():].strip() + "\n"


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(str(match.group(1)).replace(",", ""))
    except ValueError:
        return None


def _first_value(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(str(match.group(1)).strip().split())


def _extract_report_facts(text: str) -> dict[str, Any]:
    lower = text.lower()
    facts: dict[str, Any] = {
        "is_juvenile_mechanism": "juvenile hair" in lower and "multi-omics mechanism" in lower,
        "n_wgs": _first_int(r"\bn_wgs_samples\s*\|\s*([0-9,]+)", text),
        "n_h5ad": _first_int(r"\bn_h5ad_files\s*\|\s*([0-9,]+)", text),
        "n_existing_h5ad": _first_int(r"\bn_h5ad_existing\s*\|\s*([0-9,]+)", text),
        "n_candidates": _first_int(r"\bn_candidate_variants\s*\|\s*([0-9,]+)", text),
        "n_tfbs": _first_int(r"\bn_tfbs_candidates\s*\|\s*([0-9,]+)", text),
        "n_prioritized": _first_int(r"\bn_prioritized_hypotheses\s*\|\s*([0-9,]+)", text),
        "case_cells": _first_int(r"\bcase_manifest_cells\s*\|\s*([0-9,]+)", text),
        "control_cells": _first_int(r"\bcontrol_manifest_cells\s*\|\s*([0-9,]+)", text),
        "n_juvenile_donors": _first_int(r"\bn_juvenile_donors\s*\|\s*([0-9,]+)", text),
        "n_spatial": _first_int(r"\bn_spatial_files\s*\|\s*([0-9,]+)", text),
        "n_scatac": _first_int(r"\bn_scatac_files\s*\|\s*([0-9,]+)", text),
        "n_genes": _first_int(r"\bn_genes\s*\|\s*([0-9,]+)", text),
    }
    phenotype_counts = _first_value(r"\bwgs_phenotype_counts\s*\|\s*(\{[^|\n]+\})", text)
    if not phenotype_counts:
        phenotype_counts = _first_value(r"\bwgs_phenotypes\s*\|\s*(\{[^|\n]+\})", text)
    facts["phenotype_counts"] = phenotype_counts
    candidate_genes = sorted(set(re.findall(r"'gene': '([A-Za-z0-9-]+)'", text)))
    if not candidate_genes:
        candidate_genes = sorted(set(re.findall(r"\b(TYR|OCA2|SLC45A2|MC1R|HLA|NLRP1|PAX3|SOX10|MITF|TYRP1|DCT|IRF4|TNF|IFNG|CXCL10)\b", text)))
    facts["candidate_genes"] = candidate_genes[:20]
    facts["has_peak_coordinates"] = "peak_coordinate_columns_detected" in text or "| peak_coordinate_columns |" in text
    facts["has_spatial_coordinates"] = "obsm_keys" in text and "spatial" in lower
    facts["matrix_deferred"] = "matrix_deferred" in lower or "requires_matrix_scan" in lower
    facts["fallback_variants"] = "curated_locus_fallback" in lower
    facts["motif_fallback"] = "motif_prior_fallback" in lower
    return facts


def _compact_int(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "not available"


def _build_juvenile_academic_front(title: str, facts: dict[str, Any]) -> str:
    n_wgs = _compact_int(facts.get("n_wgs"))
    n_h5ad = _compact_int(facts.get("n_h5ad"))
    n_candidates = _compact_int(facts.get("n_candidates"))
    n_tfbs = _compact_int(facts.get("n_tfbs"))
    n_prioritized = _compact_int(facts.get("n_prioritized"))
    j_donors = _compact_int(facts.get("n_juvenile_donors"))
    case_cells = _compact_int(facts.get("case_cells"))
    ctrl_cells = _compact_int(facts.get("control_cells"))
    genes = ", ".join(facts.get("candidate_genes") or []) or "candidate pigmentation and immune loci"
    variant_basis = (
        "Because no upstream genome-wide association table was present in the session, the WGS step used curated pigmentation and immune loci as explicit hypothesis anchors."
        if facts.get("fallback_variants")
        else "The WGS step used session variant results as the source of candidate loci."
    )
    motif_basis = (
        "TF binding scores are motif-prior hypotheses rather than sequence-level FIMO or motifbreakR calls."
        if facts.get("motif_fallback")
        else "TF binding evidence was read from configured motif resources."
    )
    matrix_boundary = (
        "Peak-by-cell and gene-by-cell matrix contrasts were not treated as completed differential tests; they define the next quantitative validation layer."
        if facts.get("matrix_deferred")
        else "Matrix-level accessibility and expression summaries were available for interpretation."
    )
    return (
        f"# {title}\n\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "**Report type:** Polished academic report\n\n"
        "---\n\n"
        "## Abstract\n\n"
        "This analysis asked whether Juvenile hair whitening can be connected to a donor-level chain from inherited variation to regulatory context, "
        "chromatin accessibility, gene expression, and spatial tissue organization. The run linked WGS donors with BWhair Stereo-seq, scRNA-seq and scATAC-seq metadata, "
        f"covering {n_wgs} WGS samples and {n_h5ad} h5ad files. The primary interpretation is exploratory: it ranks hypotheses for follow-up rather than claiming causal variants.\n\n"
        f"The workflow prioritized {n_candidates} candidate variant or locus anchors, generated {n_tfbs} TF-binding hypotheses, and ranked {n_prioritized} multi-omics mechanism hypotheses. "
        f"Juvenile white-versus-black hair context was available for {j_donors} Juvenile donors, with {case_cells} white-hair manifest cells and {ctrl_cells} black-hair manifest cells in the relevant summaries. "
        "The strongest current conclusion is that the dataset is ready for targeted validation around pigmentation and immune genes, while full causal interpretation requires sequence-level motif scoring, exact peak overlap, and cell-type-resolved matrix contrasts.\n\n"
        "## Introduction\n\n"
        "Juvenile hair whitening is biologically plausible as a combined melanocyte, immune and regulatory phenotype. WGS can nominate inherited loci, but mechanism requires testing whether those loci fall in regulatory sequence, alter transcription-factor binding, change accessibility or expression in relevant cell states, and correspond to altered spatial organization in hair tissue.\n\n"
        "The available BWhair design is useful for this question because WGS is keyed by donor and the h5ad modalities retain donor, hair-state, age, sex and sample metadata. This makes the analysis a linked hypothesis-generation study, not a definitive association study.\n\n"
        "## Methods\n\n"
        f"WGS and donor linkage were used to define the Juvenile hair-whitening contrast and candidate loci. {variant_basis} The candidate genes/loci represented in the current output include {genes}.\n\n"
        f"Regulatory annotation mapped candidate loci to nearby pigmentation, immune and HLA context. {motif_basis} scATAC was assessed for peak-coordinate availability and white-versus-black hair accessibility readiness. scRNA was assessed for candidate-gene expression readiness and cell-type metadata. Spatial analysis inspected Stereo-seq coordinate and cell-type fields before neighbor-interaction testing.\n\n"
        "All results were interpreted with small-sample and multimodal-readiness limits. Senile_White and Vitiligo_White samples were retained for cohort context, but Juvenile hair-state follow-up used donor-linked BWhair metadata when available.\n\n"
        "## Results\n\n"
        "### Cohort and Data Readiness\n\n"
        f"The run identified {n_wgs} WGS samples and {n_h5ad} h5ad files. Phenotype counts were {facts.get('phenotype_counts') or 'reported in the source manifest'}. "
        "All reported modality conclusions should be read as donor-linked aggregate summaries unless a section states that a matrix-level test was completed.\n\n"
        "### Task 1: Genomic Candidate Variants\n\n"
        f"The WGS module produced {n_candidates} candidate variant or locus anchors. These should be treated as prioritized loci for follow-up, not as genome-wide significant discoveries. "
        "Candidate genes emphasize melanogenesis, melanocyte biology, antigen presentation and immune chemotaxis.\n\n"
        "### Task 2: Epigenomic Regulatory Context\n\n"
        f"The regulatory and TF-binding steps produced {n_tfbs} TF-binding hypothesis rows. "
        f"{motif_basis} scATAC peak metadata was {'detected' if facts.get('has_peak_coordinates') else 'not sufficient for exact interval overlap'}, "
        f"and white-versus-black Juvenile hair accessibility context contained {case_cells} versus {ctrl_cells} manifest cells. "
        f"{matrix_boundary}\n\n"
        "### Task 3: Accessibility and Expression Coupling\n\n"
        f"Expression readiness was evaluated for { _compact_int(facts.get('n_genes')) } variant-linked genes. The present evidence links candidate loci to genes and metadata fields, but it does not yet assign a quantitative direction for peak accessibility or expression change unless the raw report section below states otherwise.\n\n"
        "### Task 4: Spatial Localization and Cell-Cell Interaction\n\n"
        "Stereo-seq files contained spatial metadata suitable for readiness assessment. Spatial interaction results are currently interpreted as neighbor-graph readiness unless coordinate-derived interaction statistics are reported in the detailed tables.\n\n"
        "### Integrated Mechanism Ranking\n\n"
        f"The final ranking retained {n_prioritized} mechanism hypotheses. Evidence scores combine candidate-locus priority, regulatory context, TF priors, scATAC metadata, expression readiness and spatial readiness. These scores rank follow-up work; they are not calibrated probabilities of causality.\n\n"
        "## Discussion\n\n"
        "The analysis supports a practical follow-up strategy: prioritize pigmentation and immune loci, verify whether each candidate lies in an accessible peak, run sequence-level motif disruption, and test per-cell-type accessibility-expression coupling in matched Juvenile white and black hair samples. The current evidence is strongest for data readiness and hypothesis prioritization, and weakest for causal direction.\n\n"
        "## Limitations\n\n"
        "- Small donor counts limit genome-wide association power and make nominal signals unstable.\n"
        "- Curated locus fallback is hypothesis generation, not de novo WGS discovery.\n"
        "- Motif-prior TF disruption requires reference sequence and a configured motif scanner for quantitative delta scores.\n"
        "- scATAC/scRNA/spatial claims require backed or chunked matrix extraction and explicit cell-type models before they can be interpreted as differential molecular effects.\n\n"
        "## Conclusion\n\n"
        "BioBank Agent can execute the Juvenile hair-whitening multi-omics trajectory end to end with graceful degradation. The current output is sufficient to define candidate mechanisms and missing validation steps, but not sufficient to claim specific causal variants or cell-type effects.\n\n"
        "## Detailed Evidence Tables\n\n"
    )


def _demote_existing_front_matter(text: str) -> str:
    text = re.sub(r"^#\s+.+?\n+", "", text, count=1)
    text = re.sub(r"(?s)^\*\*Date:\*\*.*?---\s*", "", text, count=1)
    text = re.sub(r"(?s)^>\s+\*\*Key Findings\*\*.*?\n---\s*", "", text, count=1)
    text = re.sub(r"(?s)^## Executive Summary\s+This report summarizes.+?\n(?=## |\Z)", "", text, count=1)
    text = re.sub(r"(?s)^## Study Status\s+\| Field \| Status \|.*?(?=\n## |\Z)", "", text, count=1)
    return text.lstrip()


def _rewrite_noisy_section(section: str) -> str:
    lines = section.splitlines()
    if not lines:
        return section
    header = lines[0]
    body = "\n".join(lines[1:]).strip()
    if not body:
        return section
    summary_sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if not sentence:
            continue
        if any(token in sentence for token in ("Recorded quantitative outputs", "{'", "[{'", "np.float", "not recorded")):
            continue
        if sentence.startswith("|"):
            continue
        summary_sentences.append(sentence.strip())
        if len(summary_sentences) >= 3:
            break
    if not summary_sentences:
        compact = _summarize_metrics_from_table(body)
        summary_sentences = [compact] if compact else ["Detailed output is retained in the execution appendix and linked result tables."]
    return f"{header}\n\n" + " ".join(summary_sentences).strip() + "\n"


def _summarize_metrics_from_table(text: str) -> str:
    metrics = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0].lower() in {"metric", "field", "---"}:
            continue
        value = cells[1]
        if not cells[0].strip("- ") or not value.strip("- "):
            continue
        if len(value) > 80 or value.startswith(("{", "[")):
            continue
        if cells[0] and value:
            metrics.append(f"{cells[0]}={value}")
        if len(metrics) >= 5:
            break
    if not metrics:
        return ""
    return "Key recorded metrics: " + "; ".join(metrics) + "."


def _clean_main_body(text: str) -> str:
    text = text.replace("AUC=not recorded; 95% CI=not recorded", "predictive modelling was not part of the interpretable result set")
    text = re.sub(r"\b([A-Z][A-Za-z0-9/& -]*?\bCohort)\s+cohort\b", r"\1", text)
    text = re.sub(r"\bcohort\s+cohort\b", "cohort", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvisualization\s+visualization\b", "visualization", text, flags=re.IGNORECASE)
    text = re.sub(r"Recorded quantitative outputs include [^\n]+.\n\n", "", text)
    text = re.sub(r"\| [A-Za-z0-9_]+ \| \{[^|\n]{180,}\} \|\n", "", text)
    text = re.sub(r"\| [A-Za-z0-9_]+ \| \[\{[^|\n]{180,}\] \|\n", "", text)
    text = re.sub(r"\| [A-Za-z0-9_]+ \| \[[^|\n]{180,}\] \|\n", "", text)
    text = re.sub(r"\| [A-Za-z0-9_]+ \| not recorded \|\n", "", text)
    text = re.sub(r"\bnan\b", "missing", text)
    for old, new in AI_LIKE_REPLACEMENTS.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    sections = re.split(r"(?m)(?=^##\s+\d+\.\s+)", text)
    if len(sections) <= 1:
        return text.strip() + "\n"
    cleaned = [sections[0].strip()]
    for section in sections[1:]:
        if len(section) > 5000 or "{'" in section or "[{'" in section:
            cleaned.append(_rewrite_noisy_section(section))
        else:
            cleaned.append(section.strip() + "\n")
    return "\n\n".join(part for part in cleaned if part.strip()).strip() + "\n"


def _append_execution_appendix(main: str, original: str, appendix: str) -> str:
    if appendix:
        return main.rstrip() + "\n\n" + appendix.strip() + "\n"
    raw_excerpt = original.strip()
    if len(raw_excerpt) > 12000:
        raw_excerpt = raw_excerpt[:12000].rstrip() + "\n\n[Execution appendix truncated in polished report; see report_raw.md for full original output.]"
    return (
        main.rstrip()
        + "\n\n## Execution Appendix\n\n"
        + "### Analysis Record Inventory\n\n"
        + "The polished main text suppresses verbose dictionaries and logs. The original machine-readable report is preserved as `report_raw.md`; a bounded excerpt is kept below for provenance.\n\n"
        + "<details><summary>Original report excerpt</summary>\n\n"
        + "```text\n"
        + raw_excerpt.replace("```", "'''")
        + "\n```\n\n</details>\n"
    )


def _validate_figure_links(text: str, report_path: Path) -> list[str]:
    missing: list[str] = []
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if link.startswith(("http://", "https://", "data:")):
            continue
        clean = link.split("#", 1)[0].split("?", 1)[0]
        target = Path(clean)
        if not target.is_absolute():
            target = report_path.parent / target
        if not target.exists():
            missing.append(link)
    return sorted(set(missing))


def _quality_checks(text: str, report_path: Path) -> dict[str, Any]:
    main, _ = _split_main_appendix(text)
    lower_main = main.lower()
    forbidden = [p for p in FORBIDDEN_MAIN_BODY_PATTERNS if p.lower() in lower_main]
    return {
        "has_abstract": "## abstract" in text.lower(),
        "has_methods": "## methods" in text.lower(),
        "has_results": "## results" in text.lower(),
        "has_limitations": "## limitations" in text.lower(),
        "has_conclusion": "## conclusion" in text.lower(),
        "has_execution_appendix": "## execution appendix" in text.lower(),
        "no_forbidden_main_body_patterns": not forbidden,
        "forbidden_main_body_patterns": forbidden,
        "figure_links_resolvable": not _validate_figure_links(text, report_path),
        "missing_figure_links": _validate_figure_links(text, report_path),
    }


def _derive_title(text: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    if match:
        return match.group(1).strip()
    return fallback or "Biobank Analysis Report"


def polish_markdown_report(
    report_path: str | Path,
    *,
    output_path: str | Path | None = None,
    mode: str = "academic",
    strict: bool = True,
    preserve_raw: bool = True,
) -> dict[str, Any]:
    """Polish a Markdown report and return artifact/quality metadata."""
    src = Path(report_path).expanduser()
    original = _read_text(src)
    dest = Path(output_path).expanduser() if output_path else src
    raw_path = dest.with_name(dest.stem + "_raw.md")
    if preserve_raw and src.exists():
        if src.resolve() == dest.resolve():
            if not raw_path.exists():
                shutil.copy2(src, raw_path)
        elif not raw_path.exists():
            shutil.copy2(src, raw_path)

    main, appendix = _split_main_appendix(original)
    facts = _extract_report_facts(original)
    title = _derive_title(original, dest.stem.replace("_", " ").title())
    if mode == "academic" and facts.get("is_juvenile_mechanism"):
        body = _build_juvenile_academic_front(title, facts)
        body += _clean_main_body(_demote_existing_front_matter(main))
    else:
        body = _clean_main_body(main)
        if "## abstract" not in body.lower():
            body = _insert_generic_abstract_after_executive_findings(body, title)
    polished = _append_execution_appendix(body, original, appendix)
    checks = _quality_checks(polished, dest)
    warnings = []
    if checks["forbidden_main_body_patterns"]:
        warnings.append("Forbidden machine-output patterns remain in the main body: " + ", ".join(checks["forbidden_main_body_patterns"]))
    if checks["missing_figure_links"]:
        warnings.append("Missing figure links: " + ", ".join(checks["missing_figure_links"]))
    _write_text(dest, polished)
    quality_path = dest.with_name(dest.stem + "_polish_quality.json")
    quality = {
        "report_path": str(dest),
        "raw_report_path": str(raw_path) if raw_path.exists() else "",
        "mode": mode,
        "strict": bool(strict),
        "polished": True,
        "checks": checks,
        "warnings": warnings,
    }
    _write_text(quality_path, json.dumps(quality, indent=2, ensure_ascii=False))
    return {
        "markdown": str(dest),
        "raw_markdown": str(raw_path) if raw_path.exists() else "",
        "quality_json": str(quality_path),
        "polished": True,
        "polisher_warnings": warnings,
        "quality_checks": checks,
        "error": "; ".join(warnings) if strict and warnings else "",
    }


@skill(
    name="academic_report_polisher",
    description=(
        "Polish AI- or pipeline-generated technical, biomedical, bioinformatics, WGS/GWAS, omics, "
        "and academic Markdown reports into concise, human-readable senior-expert reports."
    ),
    parameters={
        "report_path": {"type": "string", "description": "Markdown report path to polish"},
        "output_path": {"type": "string", "description": "Optional output Markdown path; defaults to in-place", "default": ""},
        "mode": {"type": "string", "description": "Polishing mode", "default": "academic"},
        "strict": {"type": "boolean", "description": "Return an error if quality checks fail", "default": True},
    },
    required=["report_path"],
)
def academic_report_polisher(
    report_path: str,
    output_path: str = "",
    mode: str = "academic",
    strict: bool = True,
    *,
    ctx=None,
) -> dict:
    result = polish_markdown_report(
        report_path,
        output_path=output_path or None,
        mode=mode,
        strict=bool(strict),
    )
    if ctx is not None and hasattr(ctx, "state"):
        try:
            ctx.state.custom_data["academic_report_polisher"] = result
        except Exception:
            pass
    return result


def _insert_generic_abstract_after_executive_findings(text: str, title: str) -> str:
    """Add a compact abstract without moving executive findings out of the lead slot."""
    abstract = (
        "## Abstract\n\n"
        "This report presents the completed analysis with emphasis on interpretable results, methods, limitations and reproducibility. "
        "Verbose execution details are retained in the appendix.\n\n"
    )
    body = text
    if not re.search(r"(?m)^#\s+", body):
        body = f"# {title}\n\n" + body.lstrip()
    executive = re.search(r"(?s)(>\s+\*\*Executive Findings\*\*.*?\n---\s*)", body)
    if executive:
        return body[: executive.end()] + "\n" + abstract + body[executive.end():].lstrip()
    title_match = re.search(r"(?m)^#\s+.+?\n+", body)
    if title_match:
        return body[: title_match.end()] + "\n" + abstract + body[title_match.end():].lstrip()
    return f"# {title}\n\n{abstract}{body.lstrip()}"
