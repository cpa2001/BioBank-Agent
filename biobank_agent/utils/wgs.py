"""Shared WGS workflow helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


WGS_RESULT_DIRS = {
    "cohort": "00_Cohort",
    "qc": "01_QC",
    "annotation": "02_Annotation",
    "popgen": "03_PopGen",
    "association": "04_Association",
    "enrichment": "05_Enrichment",
}


def wgs_results_dir(ctx: Any, module: str) -> Path:
    """Return the run-local WGS result subdirectory for a workflow module."""
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        settings = getattr(ctx, "settings", None)
        report_dir = getattr(settings, "reports_dir", Path("./reports")) if settings else Path("./reports")
    subdir = WGS_RESULT_DIRS.get(module, module)
    out = Path(report_dir) / "results" / subdir
    out.mkdir(parents=True, exist_ok=True)
    if module == "popgen":
        alias = Path(report_dir) / "results_03_PopGen"
        if not alias.exists():
            try:
                alias.symlink_to(out, target_is_directory=True)
            except OSError:
                alias.mkdir(parents=True, exist_ok=True)
    return out


def external_tool_log_path(ctx: Any, report_dir: Path, stem: str) -> Path:
    """Durable per-tool log path under the WGS module result directory."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem or "external_tool")).strip("._")
    if not safe:
        safe = "external_tool"
    call_id = str(getattr(ctx, "tool_call_id", "") or "").strip()
    if call_id:
        safe = f"{safe}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', call_id)}"
    log_dir = Path(report_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{safe}.log"


def find_executable(*names: str) -> str:
    """Return the first executable path found in PATH or the active prefix."""
    prefix = Path(os.sys.prefix) / "bin"
    for name in names:
        found = shutil.which(name)
        if found:
            return found
        candidate = prefix / name
        if candidate.is_file():
            return str(candidate)
    return ""


def package_available(name: str) -> bool:
    """Return whether a Python package can be imported without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def run_external(
    args: list[str],
    *,
    timeout: int = 1800,
    cwd: str | Path | None = None,
    stdout_path: str | Path | None = None,
    settings: Any = None,
    line_sink: Any = None,
    log_path: str | Path | None = None,
) -> dict:
    """Run an external bioinformatics tool and return a compact audit record.

    ``timeout`` follows the project convention: a positive value is honored
    verbatim; ``0`` (or negative) means "auto" — scale to the tool's known
    baseline (plink/gatk/bcftools/...) via :mod:`biobank_agent.utils.exec_policy`.
    The historical default (1800s) is preserved for existing callers.

    When ``line_sink`` or ``log_path`` is provided (and no ``stdout_path`` redirect),
    the command is run via the streaming substrate so stdout/stderr can be surfaced
    live and persisted (issue #2). Callers that pass neither keep the original
    capture-and-return behavior unchanged.
    """
    from biobank_agent.utils.exec_policy import resolve_timeout

    # Foreground resolution always yields a concrete int (never None).
    timeout = int(resolve_timeout(argv=args, override=timeout, settings=settings) or 1800)

    def _write_audit_log(payload: dict) -> None:
        if log_path is None:
            return
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "$ " + " ".join(str(x) for x in args),
                f"returncode={payload.get('returncode')}",
                "",
                "[stdout]",
                str(payload.get("stdout") or ""),
                "",
                "[stderr]",
                str(payload.get("stderr") or ""),
            ]
            path.write_text("\n".join(lines), encoding="utf-8")
            payload["log_path"] = str(path)
        except Exception as exc:
            payload["log_error"] = str(exc)

    # Opt-in streaming path (no behavior change for existing callers that pass
    # neither line_sink nor log_path, nor a stdout_path redirect).
    if stdout_path is None and (line_sink is not None or log_path is not None):
        from biobank_agent.runtime.proc import run_streaming

        result = run_streaming(
            args, cwd=cwd, timeout=timeout, line_sink=line_sink,
            log_path=log_path, tail_chars=5000,
        )
        return {
            "cmd": args,
            "returncode": result.returncode,
            "stdout": (result.stdout_tail or "")[-5000:],
            "stderr": (result.stderr_tail or "")[-5000:],
            "ok": result.ok,
            "log_path": result.log_path,
        }

    stdout_handle = None
    try:
        if stdout_path is not None:
            out_path = Path(stdout_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = out_path.open("w", encoding="utf-8")
            completed = subprocess.run(
                args,
                stdout=stdout_handle,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd is not None else None,
            )
            stdout_text = f"<redirected:{out_path}>"
        else:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd is not None else None,
            )
            stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        payload = {
            "cmd": args,
            "returncode": completed.returncode,
            "stdout": stdout_text[-5000:],
            "stderr": stderr_text[-5000:],
            "ok": completed.returncode == 0,
        }
        _write_audit_log(payload)
        return payload
    except subprocess.TimeoutExpired as exc:
        payload = {
            "cmd": args,
            "returncode": None,
            "stdout": str(exc.stdout or "")[-5000:],
            "stderr": f"Timed out after {timeout}s. {str(exc.stderr or '')}"[-5000:],
            "ok": False,
        }
        _write_audit_log(payload)
        return payload
    except OSError as exc:
        payload = {
            "cmd": args,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc)[-5000:],
            "ok": False,
        }
        _write_audit_log(payload)
        return payload
    finally:
        if stdout_handle is not None:
            stdout_handle.close()


def external_cache_dir() -> Path | None:
    """Return optional cache directory for deterministic external tool outputs."""
    value = (
        os.getenv("WGS_EXTERNAL_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_EXTERNAL_CACHE_DIR", "").strip()
        or os.getenv("WGS_VCF_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_VCF_CACHE_DIR", "").strip()
    )
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_signature(path: str | Path, *, include_path: bool = True) -> dict[str, Any]:
    """Return a stable-enough signature for cache invalidation."""
    p = Path(path)
    try:
        stat = p.stat()
        out: dict[str, Any] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if include_path:
            out["path"] = str(p.resolve())
        return out
    except OSError:
        return {"path": str(p), "missing": True} if include_path else {"missing": True}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def link_or_copy(src: str | Path, dst: str | Path) -> None:
    """Hard-link when possible, copy otherwise."""
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    dst_p.unlink(missing_ok=True)
    try:
        os.link(src_p, dst_p)
    except OSError:
        shutil.copy2(src_p, dst_p)


def tool_version(*args: str, timeout: int = 30) -> str:
    """Return the first line of a tool version/help command, best effort."""
    if not args:
        return ""
    result = run_external(list(args), timeout=timeout)
    text = (result.get("stdout") or result.get("stderr") or "").strip()
    return text.splitlines()[0] if text else ""


def command_supported(executable: str, *args: str, timeout: int = 30) -> bool:
    """Return whether a command invocation exits successfully."""
    if not executable:
        return False
    result = run_external([executable, *args], timeout=timeout)
    return bool(result.get("ok"))


def snpeff_database_ready(genome: str = "hg38") -> bool:
    """Return whether a SnpEff genome database appears to be installed locally."""
    snpeff = find_executable("snpEff")
    roots: list[Path] = []
    if snpeff:
        roots.append(Path(snpeff).resolve().parent)
    roots.extend(Path(sys.prefix).glob("share/snpeff*"))
    roots.append(Path(sys.prefix))

    for root in roots:
        for data_dir in (root / "data" / genome, root / "share" / "snpeff" / "data" / genome):
            predictor = data_dir / "snpEffectPredictor.bin"
            if predictor.exists():
                return True
    return False


def vep_cache_ready(assembly: str = "GRCh38", species: str = "homo_sapiens") -> bool:
    """Return whether an Ensembl VEP cache for a species/assembly is visible."""
    roots = []
    for env in ("VEP_CACHE", "VEP_CACHE_DIR"):
        value = os.getenv(env, "").strip()
        if value:
            roots.append(Path(value).expanduser())
    roots.extend([
        Path.home() / ".vep",
        Path(sys.prefix) / "share" / "ensembl-vep",
        Path(sys.prefix) / "vep",
    ])
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.glob(f"{species}/*{assembly}*"):
            if candidate.is_dir():
                return True
    return False


def wgs_environment_status() -> dict:
    """Inspect local WGS dependencies without mutating the environment."""
    packages = {
        "cyvcf2": package_available("cyvcf2"),
        "pysam": package_available("pysam"),
        "pandas": package_available("pandas"),
        "openpyxl": package_available("openpyxl"),
        "sklearn": package_available("sklearn"),
        "scipy": package_available("scipy"),
        "matplotlib": package_available("matplotlib"),
        "seaborn": package_available("seaborn"),
        "gseapy": package_available("gseapy"),
    }
    executables = {
        "bcftools": find_executable("bcftools"),
        "tabix": find_executable("tabix"),
        "plink2": find_executable("plink2"),
        "plink": find_executable("plink"),
        "vep": find_executable("vep"),
        "snpEff": find_executable("snpEff"),
        "table_annovar.pl": find_executable("table_annovar.pl"),
        "Rscript": find_executable("Rscript"),
    }

    annotation_databases = {
        "snpeff_hg38": bool(executables["snpEff"] and snpeff_database_ready("hg38")),
        "snpeff_GRCh38_115": bool(executables["snpEff"] and snpeff_database_ready("GRCh38.115")),
        "vep_GRCh38_cache": bool(executables["vep"] and vep_cache_ready("GRCh38")),
        "annovar_available": bool(executables["table_annovar.pl"]),
    }
    has_annotation = bool(
        annotation_databases["snpeff_hg38"]
        or annotation_databases["snpeff_GRCh38_115"]
        or annotation_databases["vep_GRCh38_cache"]
        or annotation_databases["annovar_available"]
    )
    has_plink = bool(executables["plink2"] or executables["plink"])
    has_enrichment = bool(packages["gseapy"] or executables["Rscript"])
    bcftools_version_probe = (
        run_external([executables["bcftools"], "--version"], timeout=30)
        if executables["bcftools"] else {"ok": False, "stdout": "", "stderr": ""}
    )
    bcftools_merge_probe = (
        run_external([executables["bcftools"], "merge"], timeout=30)
        if executables["bcftools"] else {"ok": False, "stdout": "", "stderr": ""}
    )
    merge_text = f"{bcftools_merge_probe.get('stdout', '')}\n{bcftools_merge_probe.get('stderr', '')}"
    bcftools_modern = bool(
        bcftools_version_probe.get("ok")
        and "bcftools merge" in merge_text
    )
    has_core = bool(packages["cyvcf2"] and packages["pandas"] and bcftools_modern and executables["tabix"])

    modules = {
        "vcf_reading": "READY" if packages["cyvcf2"] and packages["pandas"] else "BLOCKED",
        "excel_manifest": "READY" if packages["openpyxl"] else "PARTIAL",
        "qc_merge": "READY" if has_core else "BLOCKED",
        "exploratory_association": "READY" if has_core and packages["scipy"] else "BLOCKED",
        "standard_gwas": "READY" if has_plink else "PARTIAL",
        "standard_annotation": "READY" if has_annotation else "PARTIAL",
        "standard_enrichment": "READY" if has_enrichment else "PARTIAL",
    }
    missing_for_standard = [
        name for name, ready in {
            "openpyxl": packages["openpyxl"],
            "plink2_or_plink": has_plink,
            "vep_or_snpeff_or_annovar": has_annotation,
            "gseapy_or_Rscript": has_enrichment,
        }.items() if not ready
    ]
    return {
        "packages": packages,
        "executables": executables,
        "executable_capabilities": {
            "bcftools_modern": bcftools_modern,
        },
        "annotation_databases": annotation_databases,
        "modules": modules,
        "workflow_mode": "standard" if has_core and not missing_for_standard else "exploratory",
        "missing_for_standard_workflow": missing_for_standard,
        "degradation_notes": [
            "Excel manifests require openpyxl when the input is .xlsx." if not packages["openpyxl"] else "",
            "bcftools is missing or too old; merge/index/stat operations require modern bcftools." if not bcftools_modern else "",
            "PLINK2/PLINK is not available; association is limited to built-in exploratory Fisher tests." if not has_plink else "",
            "No local SnpEff/VEP/ANNOVAR annotation database/cache is ready; annotation is limited to built-in hg38 candidate-gene coordinates." if not has_annotation else "",
            "gseapy/Rscript is not available; enrichment is limited to built-in compact gene sets." if not has_enrichment else "",
        ],
    }
