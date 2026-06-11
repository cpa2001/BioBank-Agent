"""Field-issues benchmark — regression cases for the real-world problems users hit in the field.

Each case exercises the ACTUAL mechanism that governs the behavior (no live LLM or external CLI), so a
regression is caught deterministically and offline. The seven field problems:

  1. long bioinformatics tools must not inherit the short generic timeout (and can background / be set)
  2. intermediate process must be visible (line-by-line streaming), not opaque until the end
  3. a data-format mismatch must pause and ask, not silently write a simplified replacement script
  4. a chosen working directory must actually receive the run's inputs/outputs
  5. a complex multi-step task must parse into a real plan, not fail or collapse to a trivial read
  6. WGS *results* handed in as input must not be mis-read as "run the WGS pipeline"
  7. a finished run must report WHERE its outputs were written

Plus domain-routing cases spanning UK Biobank, WGS, and Virtual Cell (multimodal) workflows.

Run: ``python -m biobank_agent eval --suite field_issues``
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from .harness import Benchmark, TestCase, TestResult


# ── individual checkers: each returns (ok: bool, detail: str), exercising the real mechanism ──

def _check_timeout(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.utils.exec_policy import LONG_TOOL_TIMEOUTS, resolve_timeout, should_background

    tool = next(iter(LONG_TOOL_TIMEOUTS))  # a genuinely long bioinformatics tool (plink/gatk/bcftools/…)
    long_t = resolve_timeout(command=f"{tool} --in a --out b", settings=None)
    short_t = resolve_timeout(command="echo hi", settings=None)
    explicit = resolve_timeout(command="echo hi", override=4242, settings=None)  # user can set it
    backgrounds = should_background(expected_s=7200, settings=None)  # a long job auto-backgrounds
    ok = bool(long_t) and long_t >= 3600 and long_t > (short_t or 0) and explicit == 4242 and backgrounds
    return ok, (f"long_tool({tool})={long_t}s generic={short_t}s explicit_override={explicit} "
                f"long_job_backgrounds={backgrounds}")


def _check_streaming(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.runtime.proc import run_streaming

    seen: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as d:
        res = run_streaming(
            ["bash", "-lc", "echo step-A; echo step-B 1>&2; echo step-C"],
            cwd=d,
            timeout=30,
            line_sink=lambda stream, line: seen.append((stream, line)),
            log_path=os.path.join(d, "run.log"),
            tail_chars=4000,
        )
    joined = " ".join(line for _, line in seen)
    ok = bool(getattr(res, "ok", False)) and "step-A" in joined and "step-C" in joined and len(seen) >= 3
    return ok, f"streamed_lines={len(seen)} ok={getattr(res, 'ok', None)} saw_intermediate={'step-A' in joined}"


def _check_pause_on_format_mismatch(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.skills.pause_and_ask import pause_and_ask

    out = pause_and_ask(
        "Your VCF lacks the AD/DP fields the PGS tool expects — reprocess upstream, or shall I proceed differently?",
        reason="input format does not match the standard tool's expected schema",
        category="data_format",
    )
    msg = str(out.get("message", "")).lower()
    ok = (out.get("awaiting_user") is True and out.get("category") == "data_format"
          and "simplified" in msg and "stop" in msg)
    return ok, f"awaiting_user={out.get('awaiting_user')} category={out.get('category')} guards_replacement={'simplified' in msg}"


def _check_working_directory(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.skills.local_exec import shell_exec

    with tempfile.TemporaryDirectory() as d:
        ctx = SimpleNamespace(workspace_root=d, settings=SimpleNamespace(project_root=d))
        out = shell_exec("echo hello > artifact.txt", cwd="", write_policy="workspace_write",
                         confirmed=True, ctx=ctx)
        landed = os.path.isfile(os.path.join(d, "artifact.txt"))
        resolved = str(out.get("cwd", ""))
        ok = landed and os.path.realpath(resolved) == os.path.realpath(d)
    return ok, f"output_in_workspace={landed} resolved_cwd_matches={os.path.realpath(resolved) == os.path.realpath(d)}"


def _check_complex_task_parsing(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.runtime.planner import is_pure_inspection_objective
    from biobank_agent.skills.goal_intent_classifier import classify_goal_intent

    goal = ("For UK Biobank type 2 diabetes (E11): assemble a cohort, assess missingness, train a "
            "calibrated predictive model on HbA1c and LDL, evaluate AUC with confidence intervals, "
            "run a statistical review, and write a dual-format report.")
    profile = classify_goal_intent(goal)
    # A complex analysis goal parses into a real profile AND is NOT collapsed to a trivial read-only path.
    ok = isinstance(profile, dict) and bool(profile.get("task_family")) and not is_pure_inspection_objective(goal)
    return ok, f"task_family={profile.get('task_family')} routed_to_full_planning={not is_pure_inspection_objective(goal)}"


def _check_wgs_results_not_rerun(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.runtime.planner import is_pure_inspection_objective

    # Handing in WGS *results* for a read/inspect must be a read — not "run the WGS pipeline".
    read_results = is_pure_inspection_objective("show the first 5 rows of /data/cohort_wgs_results.tsv")
    genuine_run = is_pure_inspection_objective("run GWAS association and burden testing on the cohort VCF")
    ok = read_results is True and genuine_run is False
    return ok, f"wgs_results_read_as_inspection={read_results} genuine_wgs_run_still_full_plan={not genuine_run}"


def _check_output_paths_reported(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.runtime.output_summary import summarize_output_paths

    steps = [
        {"report_dir": "/w/reports/run1", "report_markdown": "/w/reports/run1/report.md"},
        {"log_path": "/w/.biobank_jobs/inline/x.log"},
        {"figure_artifacts": ["/w/reports/run1/fig1.png", "/w/reports/run1/fig2.svg"]},
        {"status": "ok"},  # carries no path — must be ignored, not crash
    ]
    out = summarize_output_paths(steps)
    ok = (out["count"] >= 4 and "/w/reports/run1/report.md" in out["paths"]
          and "/w/reports/run1/fig1.png" in out["paths"] and "/w/reports/run1" in out["message"])
    return ok, f"paths_surfaced={out['count']} message_lists_dir={'/w/reports/run1' in out['message']}"


def _check_domain_ukb(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.skills.goal_intent_classifier import classify_goal_intent

    profile = classify_goal_intent("Generate a UK Biobank report for type 2 diabetes (E11) using field "
                                   "30750 (HbA1c) with public-paper references.")
    # A UKB tabular report must NOT be mis-routed into the multi-omics mechanism family.
    ok = isinstance(profile, dict) and profile.get("task_family") not in {"juvenile_hair_multiomics_mechanism"}
    return ok, f"task_family={profile.get('task_family')} modalities={profile.get('modalities')}"


def _check_domain_wgs(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.skills.goal_intent_classifier import classify_goal_intent

    profile = classify_goal_intent("Run QC, kinship/PCA, GWAS association and burden testing on the WGS VCF.")
    ok = "wgs" in (profile.get("modalities") or [])
    return ok, f"modalities={profile.get('modalities')}"


def _check_domain_virtualcell(_case: TestCase) -> tuple[bool, str]:
    from biobank_agent.skills.goal_intent_classifier import classify_goal_intent

    profile = classify_goal_intent("Integrate scRNA, scATAC and spatial Stereo-seq data to study the "
                                   "hair-whitening mechanism across modalities.")
    mods = set(profile.get("modalities") or [])
    ok = bool({"scrna", "scatac", "spatial"} & mods)
    return ok, f"task_family={profile.get('task_family')} modalities={sorted(mods)}"


# id -> (problem label, checker, human description)
_CASES: list[tuple[str, str, object, str]] = [
    ("field_timeout_long_tool", "problem_1_timeout", _check_timeout,
     "Long bioinformatics tools get an extended/auto-backgroundable timeout, not the 120s default."),
    ("field_streaming_visibility", "problem_2_monitoring", _check_streaming,
     "Running tools stream intermediate output line-by-line for real-time monitoring."),
    ("field_pause_on_format_mismatch", "problem_3_format_mismatch", _check_pause_on_format_mismatch,
     "A data-format mismatch pauses and asks instead of silently writing a simplified replacement."),
    ("field_working_directory", "problem_4_working_dir", _check_working_directory,
     "A chosen working directory actually receives the run's outputs."),
    ("field_complex_task_parsing", "problem_5_complex_parse", _check_complex_task_parsing,
     "A complex multi-step task parses into a real plan and is routed to full planning."),
    ("field_wgs_results_not_rerun", "problem_6_wgs_overdetect", _check_wgs_results_not_rerun,
     "WGS results handed in as input are read, not mis-routed into re-running the WGS pipeline."),
    ("field_output_paths_reported", "problem_7_output_paths", _check_output_paths_reported,
     "A finished run reports where its outputs were written."),
    ("domain_ukb_report_routing", "domain_ukb", _check_domain_ukb,
     "A UK Biobank tabular report routes correctly (not into the multi-omics mechanism family)."),
    ("domain_wgs_association_routing", "domain_wgs", _check_domain_wgs,
     "A WGS analysis goal is detected as the WGS modality."),
    ("domain_virtualcell_multimodal_routing", "domain_virtualcell", _check_domain_virtualcell,
     "A Virtual Cell multimodal goal is detected across single-cell/spatial modalities."),
]

_CHECKS = {cid: check for cid, _label, check, _desc in _CASES}


class FieldIssuesBenchmark(Benchmark):
    """Deterministic, offline regression suite for the seven field problems + domain routing."""

    name = "field_issues"

    def __init__(self) -> None:
        self.cases = [
            TestCase(
                id=cid,
                query=desc,
                tags=["field_issue", "complex", label],
                metadata={"problem": label, "description": desc},
            )
            for cid, label, _check, desc in _CASES
        ]

    def run_case(self, case: TestCase, agent) -> TestResult:
        """Exercise the real mechanism for this case — no live agent/LLM/CLI."""
        check = _CHECKS.get(case.id)
        if check is None:
            return TestResult(case_id=case.id, passed=False, errors=[f"no checker for {case.id}"],
                              metadata={"tags": case.tags})
        try:
            ok, detail = check(case)
        except Exception as e:  # a checker raising is itself a failure to surface
            return TestResult(case_id=case.id, passed=False, errors=[f"{type(e).__name__}: {str(e)[:300]}"],
                              metadata={"tags": case.tags})
        return TestResult(
            case_id=case.id,
            passed=bool(ok),
            actual_text=str(detail)[:500],
            errors=[] if ok else [str(detail)[:300]],
            metadata={"tags": case.tags, "problem": case.metadata.get("problem", "")},
        )

    def score(self, result: TestResult, case: TestCase) -> float:
        return 1.0 if result.passed else 0.0
