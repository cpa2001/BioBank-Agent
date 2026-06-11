"""Deterministic report-review support for evaluation benchmarks.

The helpers in this module are intentionally offline and side-effect free. They
provide a structured mentor/reviewer/engineer review layer for :class:`EvalHarness`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PRIMARY_REVIEWER = "report_review_primary"


@dataclass(frozen=True)
class RoleVerdict:
    """A deterministic role-level review decision."""

    role: str
    verdict: str
    checks: list[str]
    failures: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "verdict": self.verdict,
            "checks": list(self.checks),
            "failures": list(self.failures),
            "rationale": self.rationale,
        }


def reviewer_specs(primary_reviewer: str = DEFAULT_PRIMARY_REVIEWER, include_claude: bool = False) -> list[dict[str, Any]]:
    """Return reviewer specs with explicit role metadata.

    ``include_claude`` is retained for callsite compatibility and adds an
    optional secondary reviewer slot.
    """

    primary = (primary_reviewer or DEFAULT_PRIMARY_REVIEWER).strip().lower()
    specs: list[dict[str, Any]] = [
        {
            "reviewer": primary or DEFAULT_PRIMARY_REVIEWER,
            "skill": "report_review_primary",
            "role": "primary_engineer_reviewer",
            "optional": False,
        }
    ]
    if include_claude:
        specs.append(
            {
                "reviewer": "report_review_secondary",
                "skill": "report_review_secondary",
                "role": "optional_secondary_reviewer",
                "optional": True,
            }
        )
    return specs


def review_benchmark_artifacts(
    result: Any,
    benchmark: Any,
    *,
    primary_reviewer: str = DEFAULT_PRIMARY_REVIEWER,
    include_claude: bool = False,
) -> dict[str, Any]:
    """Build an offline artifact checklist and role verdicts for report benchmarks."""

    cases = getattr(benchmark, "cases", []) or []
    results = getattr(result, "results", []) or []
    report_results = [case_result for case_result in results if _is_report_result(case_result)]
    specs = reviewer_specs(primary_reviewer, include_claude)

    artifact_checks = _artifact_checklist(report_results)
    role_verdicts = _role_verdicts(artifact_checks)
    blocked = any(item.verdict == "BLOCK" for item in role_verdicts)
    skipped = not report_results and not _looks_like_report_benchmark(result, cases)

    return {
        "enabled": True,
        "status": "skipped" if skipped else ("blocked" if blocked else "passed"),
        "primary_reviewer": specs[0]["reviewer"],
        "include_claude": bool(include_claude),
        "reviewer_specs": specs,
        "artifact_count": len(report_results),
        "artifact_checklist": artifact_checks,
        "role_verdicts": [item.to_dict() for item in role_verdicts],
        "old_report_overwrite_ready": bool(
            artifact_checks.get("old_report_overwrite_ready", {}).get("passed", False)
        ),
    }


def artifact_review_gate_failures(artifact_review: dict[str, Any]) -> list[str]:
    """Translate deterministic artifact review failures into gate messages."""

    if not artifact_review or artifact_review.get("status") in {"skipped", "passed"}:
        return []
    failures: list[str] = []
    for verdict in artifact_review.get("role_verdicts", []) or []:
        if str(verdict.get("verdict", "")).upper() != "BLOCK":
            continue
        role = verdict.get("role", "role")
        failed = verdict.get("failures", []) or ["unknown"]
        failures.append(f"{role} role review BLOCK: {', '.join(str(item) for item in failed[:4])}")
    if not failures:
        failures.append(f"artifact review status={artifact_review.get('status')}")
    return failures


def _artifact_checklist(report_results: list[Any]) -> dict[str, dict[str, Any]]:
    per_artifact = [_artifact_facts(item) for item in report_results]

    def all_pass(field: str) -> bool:
        return bool(per_artifact) and all(bool(item.get(field)) for item in per_artifact)

    has_nature = any(item["nature_coverage"] for item in per_artifact)
    has_technical = any(item["technical_coverage"] for item in per_artifact)
    expected_formats = {item["format"] for item in per_artifact if item.get("format")}
    if expected_formats <= {"paper"}:
        nature_technical_passed = has_nature
    elif expected_formats <= {"report", "technical"}:
        nature_technical_passed = has_technical
    else:
        nature_technical_passed = has_nature and has_technical

    checks = {
        "executive_findings_upfront": {
            "passed": all_pass("executive_findings_upfront"),
            "description": "Each report leads with Abstract, Key Findings, or Executive Summary before method detail.",
            "failed_cases": _failed_cases(per_artifact, "executive_findings_upfront"),
        },
        "no_author_or_institution_block": {
            "passed": all_pass("no_author_or_institution_block"),
            "description": "Reports do not expose author, affiliation, or institution boilerplate.",
            "failed_cases": _failed_cases(per_artifact, "no_author_or_institution_block"),
        },
        "appendix_details": {
            "passed": all_pass("appendix_details"),
            "description": "Reports include appendix-like details through appendix, figures, references, methods, and governance material.",
            "failed_cases": _failed_cases(per_artifact, "appendix_details"),
        },
        "nature_and_technical_coverage": {
            "passed": bool(per_artifact) and nature_technical_passed,
            "description": "The benchmark artifacts cover the expected Nature-style or technical-report sections.",
            "failed_cases": [] if nature_technical_passed else [item["case_id"] for item in per_artifact],
            "has_nature_coverage": has_nature,
            "has_technical_coverage": has_technical,
        },
        "old_report_overwrite_ready": {
            "passed": all_pass("old_report_overwrite_ready"),
            "description": "Artifacts are concrete markdown reports with no placeholders, raw logs, secrets, or legacy-report boilerplate.",
            "failed_cases": _failed_cases(per_artifact, "old_report_overwrite_ready"),
        },
    }
    return checks


def _role_verdicts(checks: dict[str, dict[str, Any]]) -> list[RoleVerdict]:
    roles = [
        (
            "mentor",
            ["executive_findings_upfront", "nature_and_technical_coverage"],
            "Narrative structure is ready for senior scientific review.",
        ),
        (
            "reviewer",
            ["no_author_or_institution_block", "appendix_details"],
            "Publication-governance constraints are satisfied.",
        ),
        (
            "engineer",
            ["old_report_overwrite_ready"],
            "Artifact is deterministic and ready to replace the old report output.",
        ),
    ]
    verdicts: list[RoleVerdict] = []
    for role, names, ok_rationale in roles:
        failures = [name for name in names if not checks.get(name, {}).get("passed", False)]
        verdicts.append(
            RoleVerdict(
                role=role,
                verdict="BLOCK" if failures else "ALLOW",
                checks=list(names),
                failures=failures,
                rationale=ok_rationale if not failures else f"{role} review found blocking checklist gaps.",
            )
        )
    return verdicts


def _artifact_facts(case_result: Any) -> dict[str, Any]:
    metadata = getattr(case_result, "metadata", {}) or {}
    text = str(getattr(case_result, "actual_text", "") or "")
    report_path = metadata.get("report_path")
    if report_path:
        try:
            candidate = Path(str(report_path))
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")[:20000]
        except Exception:
            pass
    report_paths = metadata.get("report_paths")
    if isinstance(report_paths, dict) and report_paths:
        chunks = []
        for candidate_value in report_paths.values():
            try:
                candidate = Path(str(candidate_value))
                if candidate.exists():
                    chunks.append(candidate.read_text(encoding="utf-8")[:20000])
            except Exception:
                continue
        if chunks:
            text = "\n\n".join(chunks)

    lower = text.lower()
    first_section = lower[:1800]
    method_idx = _first_index(first_section, ["\n## methods", "\n## methodology", "\n## 1.", "\n### study population"])
    executive_idx = _first_index(first_section, ["executive findings", "key findings", "executive summary"])
    has_explicit_executive_findings = "executive findings" in first_section
    has_methods = "methods" in lower or "methodology notes" in lower or "statistical analysis" in lower
    has_figures = "figure" in lower
    has_refs = "references" in lower or "doi" in lower
    has_governance = "governance" in lower or "reproducibility" in lower or "data availability" in lower
    no_author_institution = not _has_author_or_institution_block(text)
    no_bad_tokens = all(token not in lower for token in _BAD_OVERWRITE_TOKENS)
    path_values = [report_path] if report_path else []
    if isinstance(report_paths, dict):
        path_values.extend(report_paths.values())
    path_ok = all(
        Path(str(path_value)).suffix.lower() in {".md", ".markdown"}
        for path_value in path_values
        if path_value
    )
    enough_text = len(text.strip()) >= 500

    return {
        "case_id": getattr(case_result, "case_id", "unknown"),
        "format": str(metadata.get("format") or "").lower(),
        "executive_findings_upfront": has_explicit_executive_findings and executive_idx >= 0 and (method_idx < 0 or executive_idx < method_idx),
        "no_author_or_institution_block": no_author_institution,
        "appendix_details": bool(("appendix" in lower) or (has_figures and has_refs and has_methods and has_governance)),
        "nature_coverage": all(token in lower for token in ("abstract", "methods", "results", "discussion")),
        "technical_coverage": all(token in lower for token in ("executive summary", "methodology notes"))
        or ("key findings" in lower and has_methods and has_governance),
        "old_report_overwrite_ready": bool(path_ok and enough_text and no_bad_tokens and no_author_institution),
    }


def _is_report_result(case_result: Any) -> bool:
    metadata = getattr(case_result, "metadata", {}) or {}
    tags = set(metadata.get("tags", []) or [])
    skills = set(getattr(case_result, "actual_skills", []) or [])
    return (
        "report" in tags
        or "generate_report" in skills
        or bool(metadata.get("report_path"))
        or bool(metadata.get("report_paths"))
        or str(getattr(case_result, "case_id", "")).startswith("report_")
    )


def _looks_like_report_benchmark(result: Any, cases: list[Any]) -> bool:
    name = str(getattr(result, "benchmark_name", "") or "").lower()
    if "report" in name:
        return True
    return any("report" in set(getattr(case, "tags", []) or []) for case in cases)


def _failed_cases(per_artifact: list[dict[str, Any]], field: str) -> list[str]:
    return [str(item["case_id"]) for item in per_artifact if not item.get(field)]


def _first_index(text: str, needles: list[str]) -> int:
    indexes = [text.find(needle) for needle in needles]
    indexes = [idx for idx in indexes if idx >= 0]
    return min(indexes) if indexes else -1


def _has_author_or_institution_block(text: str) -> bool:
    for raw in text.splitlines()[:80]:
        line = raw.strip().lower()
        if not line:
            continue
        if re.match(r"^#{1,4}\s*(authors?|affiliations?|institutions?)\b", line):
            return True
        if re.match(r"^\*{0,2}(authors?|affiliations?|institutions?)\*{0,2}\s*:", line):
            return True
    return False


_BAD_OVERWRITE_TOKENS = (
    "[citation_needed]",
    "references to be added",
    "results pending",
    "todo",
    "traceback",
    "returncode",
    "stdout",
    "stderr",
    "api_key",
    "authorization:",
    "old report placeholder",
    "legacy report",
    "analysis completed",
    "top finding: .",
    "? significant",
    "?%",
)
