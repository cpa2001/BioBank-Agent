#!/usr/bin/env python
"""Run repeated human-style CLI checks for VirtualCell/BWhair multimodal plans."""

from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import selectors
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SHORT_MULTIMODAL_TASKS = (
    "分析这批黑白发多组学数据",
    "做 VirtualCell BWhair h5ad 单细胞空间整合分析",
    "整合WGS和Stereo-seq scRNA scATAC数据生成报告",
)

SHORT_MECHANISM_TASKS = (
    "Juvenile hair whitening的多组学机制分析",
    "分析Juvenile hair whitening特异性遗传变异如何影响表观组转录组空间组",
    "少白头的基因组表观组转录组空间组机制分析",
)

CLARIFICATION_KEY_SEQUENCE = (
    b" \r"       # modalities: All Modalities
    b" \r"       # execution: Backed Metadata
    b" \r"       # grouping: Hair State
)
CLARIFICATION_RESPONSES = (
    ("Clarification: Modalities", b" \r"),
    ("Clarification: Execution", b" \r"),
    ("Clarification: Grouping", b" \r"),
)

MECHANISM_CLARIFICATION_RESPONSES = (
    ("Clarification: Contrast", b" \r"),
    ("Clarification: Depth", b" \r"),
    ("Clarification: Resources", b" \r"),
)

REQUIRED_SKILLS = {
    "virtualcell_data_inventory",
    "virtualcell_multimodal_link",
    "h5ad_sample_summary",
    "spatial_hair_summary",
    "singlecell_modality_summary",
    "statistical_review",
    "safety_check",
    "world_model_audit",
    "generate_report",
}

MECHANISM_REQUIRED_SKILLS = REQUIRED_SKILLS | {
    "goal_intent_classifier",
    "jh_variant_discovery",
    "regulatory_variant_annotation",
    "tf_binding_disruption",
    "scatac_peak_overlap",
    "scatac_accessibility_differential",
    "scrna_expression_differential",
    "atac_expression_coupling",
    "spatial_celltype_localization",
    "spatial_cell_interaction",
    "multiomics_mechanism_prioritization",
    "workflow_gap_detector",
    "agent_workflow_evolver",
}

OPTIONAL_WGS_CONTEXT_SKILLS = {
    "wgs_environment_check",
    "cohort_phenotype_summary",
}

REQUIRED_ARTIFACT_PATTERNS = {
    "inventory": (
        "results/00_Cohort/virtualcell_inventory.json",
        "results/00_Cohort/virtualcell_wgs_manifest.tsv",
        "results/00_Cohort/virtualcell_h5ad_manifest.tsv",
    ),
    "linkage": ("results/00_Cohort/virtualcell_multimodal_linkage.tsv",),
    "h5ad_summary": ("results/00_Cohort/h5ad_sample_summary.json",),
    "spatial_summary": ("results/00_Cohort/spatial_hair_summary_by_hair_state.tsv",),
    "singlecell_summary": (
        "results/00_Cohort/singlecell_modality_summary.tsv",
        "results/00_Cohort/singlecell_h5ad_inspection.json",
    ),
}

MECHANISM_REQUIRED_ARTIFACT_PATTERNS = {
    **REQUIRED_ARTIFACT_PATTERNS,
    "candidate_variants": ("results/06_MultiOmics/jh_candidate_variants.tsv",),
    "regulatory": ("results/06_MultiOmics/regulatory_variant_annotation.tsv",),
    "tf_binding": ("results/06_MultiOmics/tf_binding_disruption.tsv",),
    "scatac_overlap": ("results/06_MultiOmics/scatac_peak_overlap.tsv",),
    "scatac_da": ("results/06_MultiOmics/scatac_accessibility_differential.json",),
    "scrna_expression": ("results/06_MultiOmics/scrna_expression_differential.tsv",),
    "coupling": ("results/06_MultiOmics/atac_expression_coupling.tsv",),
    "spatial_localization": ("results/06_MultiOmics/spatial_celltype_localization.tsv",),
    "spatial_interaction": ("results/06_MultiOmics/spatial_cell_interaction.tsv",),
    "mechanism_ranking": ("results/06_MultiOmics/multiomics_mechanism_prioritization.tsv",),
    "workflow_gaps": ("results/06_MultiOmics/workflow_gap_detector.json",),
    "workflow_evolution": ("results/07_WorkflowEvolution/agent_workflow_evolution_audit.json",),
}

REQUIRED_REPORT_TERMS = (
    "VirtualCell",
    "BWhair",
    "multimodal",
    "WGS",
    "Stereo",
    "scRNA",
    "scATAC",
    "Donor",
)

MECHANISM_REQUIRED_REPORT_TERMS = REQUIRED_REPORT_TERMS + (
    "Juvenile",
    "mechanism",
    "TF",
    "ATAC",
    "expression",
    "spatial",
)

FORBIDDEN_REPORT_MAIN_TERMS = (
    "Recorded quantitative outputs",
    "not recorded cases; not recorded controls",
    "AUC=not recorded",
    "np.float64",
    "np.int64",
    "--------=-------",
)

REQUIRED_STDOUT_TERMS = ("Clarification", "Space", "Enter")


def _profile_contract(task: str, *, mechanism: bool) -> dict:
    """Load dynamic skill/artifact/report gates from trajectory profiles."""
    try:
        from biobank_agent.skills.trajectory_profile import match_trajectory_profile

        profile = match_trajectory_profile(task)
        if mechanism and profile.get("trajectory_id") != "juvenile_hair_multiomics_mechanism":
            profile = match_trajectory_profile("少白头的基因组表观组转录组空间组机制分析")
        if not mechanism and profile.get("trajectory_id") not in {"virtualcell_multimodal", "juvenile_hair_multiomics_mechanism"}:
            profile = match_trajectory_profile("分析这批黑白发多组学数据")
        required_skills = set(profile.get("required_skills") or [])
        artifact_contract = {
            key: tuple(paths)
            for key, paths in (profile.get("artifact_contract") or {}).items()
            if isinstance(paths, list)
        }
        report_terms = tuple(str(term) for term in (profile.get("report_terms") or []) if str(term))
        if required_skills:
            return {
                "trajectory_id": profile.get("trajectory_id", ""),
                "required_skills": required_skills,
                "artifact_patterns": artifact_contract,
                "report_terms": report_terms,
            }
    except Exception:
        pass
    return {
        "trajectory_id": "juvenile_hair_multiomics_mechanism" if mechanism else "virtualcell_multimodal",
        "required_skills": MECHANISM_REQUIRED_SKILLS if mechanism else REQUIRED_SKILLS,
        "artifact_patterns": MECHANISM_REQUIRED_ARTIFACT_PATTERNS if mechanism else REQUIRED_ARTIFACT_PATTERNS,
        "report_terms": MECHANISM_REQUIRED_REPORT_TERMS if mechanism else REQUIRED_REPORT_TERMS,
    }


def _latest_run_dir(reports_dir: Path) -> Path | None:
    if not reports_dir.exists():
        return None
    candidates = [p for p in reports_dir.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _action_graph_counts(memory_dir: Path) -> dict[str, int]:
    db = memory_dir / "action_graph.db"
    if not db.exists():
        return {"nodes": 0, "edges": 0}
    try:
        conn = sqlite3.connect(str(db))
        try:
            nodes = int(conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0])
            edges = int(conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0])
            return {"nodes": nodes, "edges": edges}
        finally:
            conn.close()
    except Exception:
        return {"nodes": 0, "edges": 0}


def _plan_summary(plans_dir: Path, required_skills: set[str] | None = None) -> dict:
    required_skills = required_skills or REQUIRED_SKILLS
    plan_files = sorted(
        [p for p in [*plans_dir.glob("*.json"), *plans_dir.glob("*.md")] if not p.name.startswith(".plan_checkpoint")],
        key=lambda p: p.stat().st_mtime,
    )
    if not plan_files:
        return {"skills": [], "missing_required_skills": sorted(required_skills)}
    path = plan_files[-1]
    text = path.read_text(encoding="utf-8", errors="replace")
    skills: list[str] = []
    if path.suffix == ".json":
        try:
            payload = json.loads(text)
            plan = payload.get("plan", payload)
            steps = plan.get("steps", []) if isinstance(plan, dict) else []
            skills = [str(step.get("skill", "")) for step in steps if isinstance(step, dict)]
        except Exception:
            skills = []
    else:
        import re

        skills = re.findall(r"`([A-Za-z0-9_]+)\(\{", text)
    return {
        "plan_file": str(path),
        "skills": skills,
        "missing_required_skills": sorted(required_skills - set(skills)),
        "missing_optional_context_skills": sorted(OPTIONAL_WGS_CONTEXT_SKILLS - set(skills)),
        "n_steps": len(skills),
    }


def _artifact_summary(
    run_dir: Path | None,
    required_patterns: dict[str, tuple[str, ...]] | None = None,
    required_report_terms: tuple[str, ...] | None = None,
) -> dict:
    required_patterns = required_patterns or REQUIRED_ARTIFACT_PATTERNS
    required_report_terms = required_report_terms or REQUIRED_REPORT_TERMS
    if run_dir is None:
        return {
            "run_dir": "",
            "missing_artifact_groups": sorted(required_patterns),
            "reports": [],
            "missing_report_terms": list(required_report_terms),
        }
    missing_groups = []
    hits: dict[str, list[str]] = {}
    for group, patterns in required_patterns.items():
        group_hits = []
        missing_patterns = []
        for pattern in patterns:
            pattern_hits = [str(p.relative_to(run_dir)) for p in run_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0]
            group_hits.extend(pattern_hits)
            if not pattern_hits:
                missing_patterns.append(pattern)
        hits[group] = sorted(set(group_hits))
        if missing_patterns:
            missing_groups.append(f"{group} ({', '.join(missing_patterns)})")
    reports = [
        p.name
        for p in (
            run_dir / "report.md",
            run_dir / "report.html",
            run_dir / "report_nature.md",
            run_dir / "report_nature.html",
            run_dir / "report_polish_quality.json",
            run_dir / "report_raw.md",
        )
        if p.exists() and p.stat().st_size > 0
    ]
    report_text = ""
    if (run_dir / "report.md").exists():
        report_text = (run_dir / "report.md").read_text(encoding="utf-8", errors="replace")
    report_lower = report_text.lower()
    missing_report_terms = [term for term in required_report_terms if term.lower() not in report_lower]
    main_text = report_text.split("## Execution Appendix", 1)[0]
    forbidden_main_terms = [term for term in FORBIDDEN_REPORT_MAIN_TERMS if term.lower() in main_text.lower()]
    polish_quality = {}
    quality_path = run_dir / "report_polish_quality.json"
    if quality_path.exists():
        try:
            polish_quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except Exception as exc:
            polish_quality = {"error": f"invalid polish quality json: {exc}"}
    return {
        "run_dir": str(run_dir),
        "artifact_hits": hits,
        "missing_artifact_groups": sorted(missing_groups),
        "reports": reports,
        "missing_report_terms": missing_report_terms,
        "forbidden_report_main_terms": forbidden_main_terms,
        "polish_quality": polish_quality,
    }


def _run_cli_with_pty(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    task: str,
    timeout_s: int,
    clarification_responses: tuple[tuple[str, bytes], ...] = CLARIFICATION_RESPONSES,
    live_stdout_path: Path | None = None,
) -> tuple[int, str, str, str]:
    pid, fd = pty.fork()
    stdin_log = f"/plan {task}\n<clarification_responses:{clarification_responses!r}>\n/plan-approve\nquit\n"
    if pid == 0:
        os.chdir(str(cwd))
        os.execvpe(cmd[0], cmd, env)

    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    output_chunks: list[bytes] = []
    live_stdout = None
    if live_stdout_path is not None:
        live_stdout_path.parent.mkdir(parents=True, exist_ok=True)
        live_stdout = live_stdout_path.open("wb")
    sent_plan = False
    sent_clarifications: set[str] = set()
    sent_approve = False
    sent_quit = False
    last_clarification_write = 0.0
    returncode = 1
    start_time = time.monotonic()
    deadline = time.monotonic() + timeout_s

    def write(data: bytes) -> None:
        os.write(fd, data)

    try:
        while time.monotonic() < deadline:
            for key, _ in sel.select(timeout=0.2):
                try:
                    chunk = os.read(key.fd, 8192)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        chunk = b""
                    else:
                        raise
                if not chunk:
                    try:
                        done_pid, status = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        done_pid, status = pid, 0
                    if done_pid == pid:
                        returncode = os.waitstatus_to_exitcode(status)
                        return returncode, b"".join(output_chunks).decode("utf-8", errors="replace"), "", stdin_log
                    continue
                output_chunks.append(chunk)
                if live_stdout is not None:
                    live_stdout.write(chunk)
                    live_stdout.flush()

            text = b"".join(output_chunks).decode("utf-8", errors="replace")
            if not sent_plan and ("biobank" in text.lower() or ">" in text or time.monotonic() - start_time > 3.0):
                write(f"/plan {task}\n".encode("utf-8"))
                sent_plan = True
            if sent_plan and "Space" in text and "Enter" in text:
                for marker, response in clarification_responses:
                    if marker in text and marker not in sent_clarifications:
                        write(response)
                        sent_clarifications.add(marker)
                        last_clarification_write = time.monotonic()
                        break
                else:
                    if (
                        len(sent_clarifications) < len(clarification_responses)
                        and time.monotonic() - last_clarification_write > 4.0
                    ):
                        write(b" \r")
                        last_clarification_write = time.monotonic()
            if sent_plan and not sent_approve and ("Awaiting your review" in text or "Plan Review" in text):
                write(b"/plan-approve\n")
                sent_approve = True
            if sent_approve and not sent_quit and ("Plan execution complete" in text or ("report.html" in text and "report.md" in text)):
                write(b"quit\n")
                sent_quit = True

            try:
                done_pid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done_pid, status = pid, 0
            if done_pid == pid:
                returncode = os.waitstatus_to_exitcode(status)
                break
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
            returncode = 124
    finally:
        if live_stdout is not None:
            live_stdout.close()
        try:
            sel.unregister(fd)
        except Exception:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
    return returncode, b"".join(output_chunks).decode("utf-8", errors="replace"), "", stdin_log


def _run_cli_noninteractive(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    task: str,
    timeout_s: int,
) -> tuple[int, str, str, str]:
    stdin_text = f"/plan {task}\n/plan-approve\nquit\n"
    completed = subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
        cwd=str(cwd),
    )
    return completed.returncode, completed.stdout or "", completed.stderr or "", stdin_text


def run_round(
    round_idx: int,
    base_dir: Path,
    timeout_s: int,
    python_cmd: str,
    cache_dir: Path | None = None,
    use_pty: bool = True,
    mechanism: bool = False,
) -> dict:
    round_dir = base_dir / f"round_{round_idx:02d}"
    reports_dir = round_dir / "reports"
    memory_dir = round_dir / "memory"
    plans_dir = round_dir / "plans"
    for path in (reports_dir, memory_dir, plans_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "REPORTS_DIR": str(reports_dir),
        "MEMORY_DIR": str(memory_dir),
        "PLANS_DIR": str(plans_dir),
        "BANK_ID": "virtualcell",
        "BIOBANK_NAME": "VirtualCell/BWhair Multimodal Cohort",
        "BIOBANK_ABBREVIATION": "VC",
        "DATA_DIR": env.get("DATA_DIR", str(Path.cwd() / "data" / "virtualcell")),
        "RAW_DIR": env.get("RAW_DIR", str(Path.cwd() / "data" / "virtualcell")),
        "SUBJECT_ID_COL": "sample_id",
        "DIAGNOSES_CODE_COL": "diag_code",
        "DEATHS_CODE_COL": "cause_code",
        "VC_WGS_VCF_DIR": env.get("VC_WGS_VCF_DIR", "/Files/ResultData/BW_WGS_vcf"),
        "VC_BWHAIR_SPATIAL_DIR": env.get("VC_BWHAIR_SPATIAL_DIR", "/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial"),
        "VC_BWHAIR_SINGLECELL_DIR": env.get("VC_BWHAIR_SINGLECELL_DIR", "/Files/ResultData/VirtualCell_BWhair/01.VirtualCell_BWhair_scRNA_scATAC"),
        "PLAN_EXTERNAL_COUNCIL_POLICY": "never",
        "PLAN_REVIEW_HOOK_MODE": "never",
        "PLAN_REVIEW_REPAIR_MODE": "never",
        "AUTO_DISCOVER_MODELS": "false",
        "MULTI_MODEL_ENABLED": "false",
        "ASYNC_RUNTIME_ENABLED": "false",
    })
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["VC_H5AD_METADATA_CACHE_DIR"] = str(cache_dir / "h5ad")
        env["DEEP_RESEARCH_CACHE_DIR"] = str(cache_dir / "deep_research")
    task_pool = SHORT_MECHANISM_TASKS if mechanism else SHORT_MULTIMODAL_TASKS
    task = task_pool[(round_idx - 1) % len(task_pool)]
    contract = _profile_contract(task, mechanism=mechanism)
    cmd = [python_cmd, "-m", "biobank_agent.cli"]
    if use_pty:
        returncode, stdout, stderr, stdin_text = _run_cli_with_pty(
            cmd,
            env=env,
            cwd=Path.cwd(),
            task=task,
            timeout_s=timeout_s,
            clarification_responses=MECHANISM_CLARIFICATION_RESPONSES if mechanism else CLARIFICATION_RESPONSES,
            live_stdout_path=round_dir / "stdout.live.txt",
        )
    else:
        returncode, stdout, stderr, stdin_text = _run_cli_noninteractive(cmd, env=env, cwd=Path.cwd(), task=task, timeout_s=timeout_s)

    (round_dir / "stdin.txt").write_text(stdin_text, encoding="utf-8")
    (round_dir / "task.txt").write_text(task + "\n", encoding="utf-8")
    (round_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (round_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")

    plan = _plan_summary(plans_dir, contract["required_skills"])
    artifacts = _artifact_summary(
        _latest_run_dir(reports_dir),
        contract["artifact_patterns"],
        contract["report_terms"],
    )
    graph = _action_graph_counts(memory_dir)
    errors = []
    out_text = f"{stdout}\n{stderr}".lower()
    if returncode != 0:
        errors.append(f"CLI exited with return code {returncode}")
    if "traceback" in out_text:
        errors.append("CLI output contains traceback")
    if use_pty:
        missing_stdout = [term for term in REQUIRED_STDOUT_TERMS if term.lower() not in out_text]
        if missing_stdout:
            errors.append("PTY interaction missing terms: " + ", ".join(missing_stdout))
    if plan["missing_required_skills"]:
        errors.append("Plan missing multimodal skills: " + ", ".join(plan["missing_required_skills"]))
    if artifacts["missing_artifact_groups"]:
        errors.append("Missing required artifacts: " + ", ".join(artifacts["missing_artifact_groups"]))
    if artifacts["missing_report_terms"]:
        errors.append("Report missing required terms: " + ", ".join(artifacts["missing_report_terms"]))
    if artifacts.get("forbidden_report_main_terms"):
        errors.append("Report main body contains machine-output terms: " + ", ".join(artifacts["forbidden_report_main_terms"]))
    quality = artifacts.get("polish_quality") or {}
    checks = quality.get("checks") or {}
    if not quality.get("polished"):
        errors.append("Missing academic report polish quality metadata")
    if checks and checks.get("no_forbidden_main_body_patterns") is False:
        errors.append("Polisher quality gate found forbidden main-body patterns")
    if checks and checks.get("figure_links_resolvable") is False:
        errors.append("Polisher quality gate found broken figure links")
    if not {"report.md", "report.html"} & set(artifacts["reports"]):
        errors.append("No non-empty Markdown/HTML report found")
    if graph["nodes"] <= 0 or graph["edges"] <= 0:
        errors.append("Action Graph has no nodes or edges")

    summary = {
        "round": round_idx,
        "ok": not errors,
        "errors": errors,
        "returncode": returncode,
        "task": task,
        "trajectory_id": contract["trajectory_id"],
        "use_pty": use_pty,
        "plan": plan,
        "artifacts": artifacts,
        "action_graph": graph,
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-pty", action="store_true")
    parser.add_argument("--cache-dir", default="", help="Shared h5ad metadata cache directory")
    parser.add_argument("--mechanism", action="store_true", help="Run Juvenile hair-whitening mechanism tasks")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir or f"reports/virtualcell_multimodal_cli_e2e_{stamp}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else base_dir / "h5ad_metadata_cache"
    summaries = []
    for idx in range(1, args.rounds + 1):
        summary = run_round(
            idx,
            base_dir,
            args.timeout_s,
            args.python,
            cache_dir=cache_dir,
            use_pty=not args.no_pty,
            mechanism=args.mechanism,
        )
        summaries.append(summary)
        print(f"round {idx:02d}: {'OK' if summary['ok'] else 'FAIL'}", flush=True)
        if not summary["ok"]:
            print(json.dumps(summary["errors"], ensure_ascii=False), flush=True)
    aggregate = {
        "rounds": args.rounds,
        "passed": sum(1 for s in summaries if s["ok"]),
        "failed": sum(1 for s in summaries if not s["ok"]),
        "use_pty": not args.no_pty,
        "mechanism": args.mechanism,
        "short_tasks": list(SHORT_MECHANISM_TASKS if args.mechanism else SHORT_MULTIMODAL_TASKS),
        "summaries": summaries,
    }
    (base_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(base_dir), "passed": aggregate["passed"], "failed": aggregate["failed"]}, ensure_ascii=False))
    return 0 if aggregate["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
