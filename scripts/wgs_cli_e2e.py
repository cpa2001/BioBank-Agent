#!/usr/bin/env python
"""Run repeated human-style CLI end-to-end checks for the WGS workflow."""

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


SHORT_WGS_TASKS = (
    "分析这批白癜风WGS数据",
    "比较青少年白癜风和白癜风组的WGS遗传差异",
    "用现有VirtualCell VCF做白癜风WGS分析",
)

CLARIFICATION_KEY_SEQUENCE = (
    b" \r"       # data source: select recommended Auto Discover
    b"\x1b[B \r"  # groups: move once, select conservative Ask If Missing
    b" \r"       # scope: select recommended Tractable WGS
)


REQUIRED_SKILLS = {
    "wgs_environment_check",
    "cohort_phenotype_summary",
    "vcf_sample_list",
    "vcf_qc",
    "vcf_annotation",
    "vcf_pca",
    "vcf_kinship",
    "vcf_association",
    "vcf_burden_test",
    "pathway_enrichment",
    "vcf_phenotype_comparison",
    "train_phenotype_model",
    "feature_importance",
    "embedding",
    "deep_research",
    "statistical_review",
    "safety_check",
    "world_model_audit",
    "generate_report",
}


REQUIRED_RESULT_DIRS = (
    "00_Cohort",
    "01_QC",
    "02_Annotation",
    "03_PopGen",
    "04_Association",
    "05_Enrichment",
)
REQUIRED_RUN_DIRS = ("results_03_PopGen",)

REQUIRED_ARTIFACT_PATTERNS = {
    "qc_figures": ("results/01_QC/vcf_qc_*.pdf", "results/01_QC/vcf_qc_*.svg"),
    "annotation_table": ("results/02_Annotation/*annotations*.tsv",),
    "popgen_figures": ("results/03_PopGen/vcf_pca_*.pdf", "results/03_PopGen/vcf_kinship_*.pdf"),
    "association_tables": ("results/04_Association/association_results.tsv", "results/04_Association/top_hits.tsv"),
    "association_figures": (
        "results/04_Association/vcf_association_manhattan.pdf",
        "results/04_Association/vcf_association_qq.pdf",
    ),
    "burden_figure": ("results/04_Association/vcf_burden_test.pdf",),
    "enrichment_table": ("results/05_Enrichment/*enrichment*.tsv",),
    "enrichment_figure": ("results/05_Enrichment/pathway_enrichment.pdf",),
    "model_tuning": ("phenotype_model_tuning_*.tsv", "phenotype_model_tuning_*.pdf"),
    "feature_importance": ("feature_importance_*.pdf", "feature_importance_*.svg"),
    "embedding": ("embedding_*.pdf", "embedding_*.svg"),
    "reproducibility_scripts": (
        "run_reproduce_wgs.sh",
        "run_reproduce_wgs.py",
        "reproducibility_manifest.json",
    ),
}

REQUIRED_REPORT_TERMS = (
    "Juvenile_White",
    "Vitiligo_White",
    "WGS",
    "QC",
    "association",
    "pathway",
    "literature",
    "reproducib",
)

REQUIRED_STDOUT_TERMS = (
    "Clarification",
    "Use",
    "Space",
    "Enter",
)


def _latest_run_dir(reports_dir: Path) -> Path | None:
    if not reports_dir.exists():
        return None
    candidates = [p for p in reports_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


def _plan_summary(plans_dir: Path) -> dict:
    plan_files = sorted(
        [
            *[p for p in plans_dir.glob("*.json") if not p.name.startswith(".")],
            *[p for p in plans_dir.glob("*.md") if not p.name.startswith(".")],
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not plan_files:
        return {"skills": [], "missing_required_skills": sorted(REQUIRED_SKILLS)}
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
        "missing_required_skills": sorted(REQUIRED_SKILLS - set(skills)),
        "n_steps": len(skills),
    }


def _artifact_summary(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {
            "run_dir": "",
            "missing_result_dirs": list(REQUIRED_RESULT_DIRS),
            "missing_run_dirs": list(REQUIRED_RUN_DIRS),
            "reports": [],
            "missing_artifact_groups": sorted(REQUIRED_ARTIFACT_PATTERNS),
            "missing_report_terms": list(REQUIRED_REPORT_TERMS),
        }
    results_dir = run_dir / "results"
    missing_dirs = [name for name in REQUIRED_RESULT_DIRS if not (results_dir / name).is_dir()]
    missing_run_dirs = [name for name in REQUIRED_RUN_DIRS if not (run_dir / name).is_dir()]
    reports = [p.name for p in (run_dir / "report.md", run_dir / "report.html", run_dir / "report_nature.md", run_dir / "report_nature.html") if p.exists() and p.stat().st_size > 0]
    missing_artifact_groups = []
    artifact_hits: dict[str, list[str]] = {}
    for name, patterns in REQUIRED_ARTIFACT_PATTERNS.items():
        hits = []
        missing_patterns = []
        for pattern in patterns:
            pattern_hits = [
                str(p.relative_to(run_dir))
                for p in run_dir.glob(pattern)
                if p.is_file() and p.stat().st_size > 0
            ]
            hits.extend(pattern_hits)
            if not pattern_hits:
                missing_patterns.append(pattern)
        artifact_hits[name] = sorted(set(hits))
        if missing_patterns:
            missing_artifact_groups.append(f"{name} ({', '.join(missing_patterns)})")
    report_text = ""
    report_md = run_dir / "report.md"
    if report_md.exists():
        report_text = report_md.read_text(encoding="utf-8", errors="replace")
    report_lower = report_text.lower()
    missing_report_terms = [
        term for term in REQUIRED_REPORT_TERMS
        if term.lower() not in report_lower
    ]
    key_artifacts = []
    for pattern in (
        "results/01_QC/*",
        "results/02_Annotation/*",
        "results/03_PopGen/*",
        "results/04_Association/*",
        "results/05_Enrichment/*",
    ):
        key_artifacts.extend(str(p.relative_to(run_dir)) for p in run_dir.glob(pattern) if p.is_file())
    return {
        "run_dir": str(run_dir),
        "missing_result_dirs": missing_dirs,
        "missing_run_dirs": missing_run_dirs,
        "missing_artifact_groups": sorted(missing_artifact_groups),
        "artifact_hits": artifact_hits,
        "missing_report_terms": missing_report_terms,
        "reports": reports,
        "n_key_artifacts": len(key_artifacts),
        "key_artifacts": key_artifacts[:80],
    }


def _run_cli_with_pty(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    task: str,
    timeout_s: int,
) -> tuple[int, str, str, str]:
    """Run the CLI in a PTY and answer clarification prompts with real key sequences."""
    pid, fd = pty.fork()
    stdin_log = f"/plan {task}\n<keys:{CLARIFICATION_KEY_SEQUENCE!r}>\n/plan-approve\nquit\n"
    if pid == 0:
        os.chdir(str(cwd))
        os.execvpe(cmd[0], cmd, env)

    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    output_chunks: list[bytes] = []
    sent_plan = False
    sent_keys = False
    sent_approve = False
    sent_quit = False
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

            text = b"".join(output_chunks).decode("utf-8", errors="replace")
            if not sent_plan and (
                "biobank" in text.lower()
                or ">" in text
                or time.monotonic() - start_time > 3.0
            ):
                write(f"/plan {task}\n".encode("utf-8"))
                sent_plan = True
            if sent_plan and not sent_keys and "Use" in text and "Space" in text and "Enter" in text:
                write(CLARIFICATION_KEY_SEQUENCE)
                sent_keys = True
            if sent_plan and not sent_approve and (
                "Awaiting your review" in text
                or "Plan Review" in text
            ):
                if not sent_keys:
                    sent_keys = True
                write(b"/plan-approve\n")
                sent_approve = True
            if sent_approve and not sent_quit and "Plan execution complete" in text:
                write(b"quit\n")
                sent_quit = True
            if sent_approve and not sent_quit and "report.html" in text and "report.md" in text:
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
) -> dict:
    round_dir = base_dir / f"round_{round_idx:02d}"
    reports_dir = round_dir / "reports"
    memory_dir = round_dir / "memory"
    plans_dir = round_dir / "plans"
    for p in (reports_dir, memory_dir, plans_dir):
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "REPORTS_DIR": str(reports_dir),
        "MEMORY_DIR": str(memory_dir),
        "PLANS_DIR": str(plans_dir),
        "BANK_ID": "virtualcell",
        "BIOBANK_NAME": "VirtualCell WGS Cohort",
        "BIOBANK_ABBREVIATION": "VC",
        "DATA_DIR": env.get("DATA_DIR", str(Path.cwd() / "data" / "virtualcell")),
        "RAW_DIR": env.get("RAW_DIR", str(Path.cwd() / "data" / "virtualcell")),
        "SUBJECT_ID_COL": "sample_id",
        "DIAGNOSES_CODE_COL": "diag_code",
        "DEATHS_CODE_COL": "cause_code",
        "VC_WGS_VCF_DIR": env.get("VC_WGS_VCF_DIR", "/Files/ResultData/BW_WGS_vcf"),
        "PLAN_EXTERNAL_COUNCIL_POLICY": "never",
        "PLAN_REVIEW_HOOK_MODE": "never",
        "PLAN_REVIEW_REPAIR_MODE": "never",
        "AUTO_DISCOVER_MODELS": "false",
        "MULTI_MODEL_ENABLED": "false",
        "ASYNC_RUNTIME_ENABLED": "false",
    })
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["WGS_VCF_CACHE_DIR"] = str(cache_dir)
    task = SHORT_WGS_TASKS[(round_idx - 1) % len(SHORT_WGS_TASKS)]

    cmd = [python_cmd, "-m", "biobank_agent.cli"]
    if use_pty:
        returncode, stdout, stderr, stdin_text = _run_cli_with_pty(
            cmd,
            env=env,
            cwd=Path.cwd(),
            task=task,
            timeout_s=timeout_s,
        )
    else:
        returncode, stdout, stderr, stdin_text = _run_cli_noninteractive(
            cmd,
            env=env,
            cwd=Path.cwd(),
            task=task,
            timeout_s=timeout_s,
        )
    (round_dir / "stdin.txt").write_text(stdin_text, encoding="utf-8")
    (round_dir / "task.txt").write_text(task + "\n", encoding="utf-8")
    (round_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (round_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")

    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )

    plan = _plan_summary(plans_dir)
    artifacts = _artifact_summary(_latest_run_dir(reports_dir))
    graph = _action_graph_counts(memory_dir)
    errors = []
    out_text = f"{completed.stdout}\n{completed.stderr}".lower()
    if completed.returncode != 0:
        errors.append(f"CLI exited with return code {completed.returncode}")
    if "traceback" in out_text:
        errors.append("CLI output contains traceback")
    if use_pty:
        missing_stdout_terms = [
            term for term in REQUIRED_STDOUT_TERMS
            if term.lower() not in out_text
        ]
        if missing_stdout_terms:
            errors.append("PTY interaction missing terms: " + ", ".join(missing_stdout_terms))
    if plan["missing_required_skills"]:
        errors.append("Plan missing WGS skills: " + ", ".join(plan["missing_required_skills"]))
    if artifacts["missing_result_dirs"]:
        errors.append("Missing result dirs: " + ", ".join(artifacts["missing_result_dirs"]))
    if artifacts.get("missing_run_dirs"):
        errors.append("Missing run dirs: " + ", ".join(artifacts["missing_run_dirs"]))
    if artifacts.get("missing_artifact_groups"):
        errors.append("Missing required artifacts: " + ", ".join(artifacts["missing_artifact_groups"]))
    if artifacts.get("missing_report_terms"):
        errors.append("Report missing required terms: " + ", ".join(artifacts["missing_report_terms"]))
    if not {"report.md", "report.html"} & set(artifacts["reports"]):
        errors.append("No non-empty Markdown/HTML report found")
    if graph["nodes"] <= 0 or graph["edges"] <= 0:
        errors.append("Action Graph has no nodes or edges")

    summary = {
        "round": round_idx,
        "ok": not errors,
        "errors": errors,
        "returncode": completed.returncode,
        "task": task,
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
    parser.add_argument("--no-pty", action="store_true", help="Use stdin piping instead of real PTY key interaction")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Shared deterministic WGS cache directory (default: <output-dir>/cache)",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir or f"reports/wgs_cli_e2e_{stamp}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else base_dir / "cache"

    summaries = []
    for idx in range(1, args.rounds + 1):
        summary = run_round(
            idx,
            base_dir,
            args.timeout_s,
            args.python,
            cache_dir=cache_dir,
            use_pty=not args.no_pty,
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
        "short_tasks": list(SHORT_WGS_TASKS),
        "summaries": summaries,
    }
    (base_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(base_dir), "passed": aggregate["passed"], "failed": aggregate["failed"]}, ensure_ascii=False))
    return 0 if aggregate["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
