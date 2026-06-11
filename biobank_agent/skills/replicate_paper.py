"""Half-automatic paper replication planning skill."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from biobank_agent.domain.reproducibility import PaperReplicator, render_replication_checklist
from biobank_agent.registry import skill


_DOI_OR_URL = re.compile(r"^(?:10\.\d{4,9}/|https?://)", re.I)


@skill(
    name="replicate_paper",
    description=(
        "Extract a paper's study design from text or a local PDF/text path and "
        "produce a review-only replication StudySpec plus proposed plan. This "
        "does not execute the replication until the user approves the plan."
    ),
    parameters={
        "source": {
            "type": "string",
            "description": "Paper text, or path to a local .pdf/.txt/.md file.",
        },
        "source_type": {
            "type": "string",
            "description": "auto, text, path, doi, url, or identifier. identifier creates a review plan without fetching.",
            "default": "auto",
        },
    },
    required=["source"],
)
def replicate_paper(source: str, source_type: str = "auto", *, ctx=None) -> dict:
    source = str(source or "").strip()
    source_type = str(source_type or "auto").strip().lower()
    if not source:
        return {"error": "source is required"}

    replicator = PaperReplicator()
    path = Path(source).expanduser()
    source_note = ""
    if source_type == "path" or (source_type == "auto" and path.exists()):
        outcome = replicator.from_path(path)
    elif source_type == "identifier":
        outcome = replicator.from_text(source, paper_path=source)
    elif source_type in {"doi", "url"} or (source_type == "auto" and _DOI_OR_URL.search(source)):
        try:
            from biobank_agent.skills.read_paper import read_paper

            paper = read_paper(paper_path_or_doi=source, focus="ukb-relevance", ctx=ctx)
            if paper.get("error"):
                outcome = replicator.from_text(source, paper_path=source)
                source_note = f"read_paper could not extract full text: {paper['error']}"
            else:
                summary = paper.get("summary", {}) or {}
                paper_text = "\n\n".join(
                    part
                    for part in (
                        str(summary.get("title") or ""),
                        str(summary.get("doi") or source),
                        str(paper.get("paper_text") or ""),
                        str(paper.get("analysis_template") or ""),
                    )
                    if part
                )
                outcome = replicator.from_text(paper_text, paper_path=source)
                if paper.get("text_truncated"):
                    source_note = (
                        "read_paper returned truncated text for planning; exact "
                        "replication still requires the execution plan to record "
                        "paper-access limitations."
                    )
                if summary.get("access_warning"):
                    source_note = str(summary["access_warning"])
        except Exception as exc:
            outcome = replicator.from_text(source, paper_path=source)
            source_note = f"read_paper failed; generated fallback plan from identifier only: {exc}"
    else:
        outcome = replicator.from_text(source, paper_path="")
    if source_note:
        outcome.notes.append(source_note)

    payload = asdict(outcome)
    payload["status"] = "AWAITING_USER_APPROVAL"
    payload["message"] = (
        "Replication design extracted. Review proposed_spec and plan before "
        "executing any cohort or modelling steps."
    )

    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is not None:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        out_path = report_path / "paper_replication_spec.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["artifact"] = str(out_path)
        checklist_path = report_path / "paper_replication_review.md"
        checklist_path.write_text(render_replication_checklist(outcome), encoding="utf-8")
        plan_path = report_path / "paper_replication_plan.json"
        plan_path.write_text(json.dumps(outcome.plan, indent=2, default=str), encoding="utf-8")
        payload["artifacts"] = {
            "spec_json": str(out_path),
            "review_markdown": str(checklist_path),
            "plan_json": str(plan_path),
        }

    return payload
