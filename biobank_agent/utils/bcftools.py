"""bcftools subprocess wrapper for VCF operations."""

from __future__ import annotations

import logging
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BCFTOOLS: Optional[str] = None


def bcftools_path() -> str:
    global _BCFTOOLS
    if _BCFTOOLS is None:
        found = shutil.which("bcftools")
        if not found:
            candidate = os.path.join(sys.prefix, "bin", "bcftools")
            if os.path.isfile(candidate):
                found = candidate
        if not found:
            raise FileNotFoundError(
                "bcftools not found. Install with: conda install -c bioconda bcftools"
            )
        _BCFTOOLS = found
    return _BCFTOOLS


def _tabix_path() -> str:
    found = shutil.which("tabix")
    if not found:
        candidate = os.path.join(sys.prefix, "bin", "tabix")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError("tabix not found")
    return found


def _path_signature(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        stat = p.stat()
        return {
            "path": str(p.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError:
        return {"path": str(p), "missing": True}


def _merge_cache_dir() -> Path | None:
    value = (
        os.getenv("WGS_MERGE_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_MERGE_CACHE_DIR", "").strip()
        or os.getenv("WGS_VCF_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_VCF_CACHE_DIR", "").strip()
    )
    if not value:
        return None
    cache_dir = Path(value).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _filter_cache_dir() -> Path | None:
    value = (
        os.getenv("WGS_FILTER_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_FILTER_CACHE_DIR", "").strip()
        or os.getenv("WGS_VCF_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_VCF_CACHE_DIR", "").strip()
    )
    if not value:
        return None
    cache_dir = Path(value).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _merge_cache_key(
    vcf_paths: list[str],
    region: Optional[str],
    missing_to_ref: bool,
) -> str:
    payload = {
        "version": 1,
        "operation": "bcftools_merge_or_single_view",
        "region": region or "",
        "missing_to_ref": bool(missing_to_ref),
        "inputs": [_path_signature(p) for p in vcf_paths],
    }
    text = repr(payload).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _filter_cache_key(
    input_path: str | Path,
    include_expr: Optional[str],
    exclude_expr: Optional[str],
    region: Optional[str],
) -> str:
    payload = {
        "version": 1,
        "operation": "bcftools_view_filter",
        "input": _path_signature(input_path),
        "include_expr": include_expr or "",
        "exclude_expr": exclude_expr or "",
        "region": region or "",
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _index_path(vcf_path: str | Path) -> Path:
    return Path(str(vcf_path) + ".tbi")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize_cached_vcf(cache_path: Path, output_path: Path) -> bool:
    cache_index = _index_path(cache_path)
    if not cache_path.exists() or not cache_index.exists():
        return False
    _link_or_copy(cache_path, output_path)
    _link_or_copy(cache_index, _index_path(output_path))
    return True


def _store_cached_vcf(output_path: Path, cache_path: Path) -> None:
    output_index = _index_path(output_path)
    if not output_path.exists() or not output_index.exists():
        return
    cache_index = _index_path(cache_path)
    if cache_path.exists() and cache_index.exists():
        return
    tmp_cache = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    tmp_index = Path(str(tmp_cache) + ".tbi")
    try:
        _link_or_copy(output_path, tmp_cache)
        _link_or_copy(output_index, tmp_index)
        os.replace(tmp_cache, cache_path)
        os.replace(tmp_index, cache_index)
    finally:
        tmp_cache.unlink(missing_ok=True)
        tmp_index.unlink(missing_ok=True)


def run_bcftools(
    args: list[str],
    timeout: int = 600,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [bcftools_path()] + args
    logger.debug("bcftools: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"bcftools failed (rc={result.returncode}): {stderr}")
    return result


def tabix_index(vcf_path: str | Path) -> None:
    vcf_path = str(vcf_path)
    subprocess.run(
        [_tabix_path(), "-p", "vcf", "-f", vcf_path],
        capture_output=True,
        text=True,
        timeout=120,
    )


def merge_vcfs(
    vcf_paths: list[str],
    output_path: str | Path,
    region: Optional[str] = None,
    missing_to_ref: bool = True,
    timeout: int = 1800,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path: Path | None = None
    cache_dir = _merge_cache_dir()
    if cache_dir is not None:
        cache_key = _merge_cache_key(vcf_paths, region, missing_to_ref)
        cache_path = cache_dir / f"merged_{cache_key}.vcf.gz"
        if _materialize_cached_vcf(cache_path, output_path):
            return output_path

    if len(vcf_paths) == 1:
        args = ["view", "-Oz", "-o", str(output_path)]
        if region:
            args.extend(["-r", region])
        args.append(vcf_paths[0])
        run_bcftools(args, timeout=timeout)
    else:
        list_file = output_path.parent / (output_path.stem + ".filelist.txt")
        list_file.write_text("\n".join(vcf_paths) + "\n")
        args = ["merge", "-Oz", "-o", str(output_path)]
        if missing_to_ref:
            args.append("-0")
        if region:
            args.extend(["-r", region])
        args.extend(["--file-list", str(list_file)])
        run_bcftools(args, timeout=timeout)
        list_file.unlink(missing_ok=True)

    tabix_index(output_path)
    if cache_path is not None:
        _store_cached_vcf(output_path, cache_path)
    return output_path


def filter_vcf(
    input_path: str | Path,
    output_path: str | Path,
    include_expr: Optional[str] = None,
    exclude_expr: Optional[str] = None,
    region: Optional[str] = None,
    timeout: int = 600,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path: Path | None = None
    cache_dir = _filter_cache_dir()
    if cache_dir is not None:
        cache_key = _filter_cache_key(input_path, include_expr, exclude_expr, region)
        cache_path = cache_dir / f"filtered_{cache_key}.vcf.gz"
        if _materialize_cached_vcf(cache_path, output_path):
            return output_path

    args = ["view", "-Oz", "-o", str(output_path)]
    if region:
        args.extend(["-r", region])
    if include_expr:
        args.extend(["-i", include_expr])
    if exclude_expr:
        args.extend(["-e", exclude_expr])
    args.append(str(input_path))
    run_bcftools(args, timeout=timeout)
    tabix_index(output_path)
    if cache_path is not None:
        _store_cached_vcf(output_path, cache_path)
    return output_path


def stats_vcf(vcf_path: str | Path) -> dict[str, Any]:
    result = run_bcftools(["stats", str(vcf_path)])
    parsed: dict[str, Any] = {
        "n_records": 0,
        "n_snps": 0,
        "n_indels": 0,
        "n_mnps": 0,
        "n_others": 0,
        "ti_tv_ratio": None,
        "n_samples": 0,
    }
    for line in result.stdout.splitlines():
        if line.startswith("SN\t"):
            parts = line.split("\t")
            if len(parts) >= 4:
                key = parts[2].strip().rstrip(":")
                val = parts[3].strip()
                if "number of records" in key:
                    parsed["n_records"] = int(val)
                elif "number of SNPs" in key:
                    parsed["n_snps"] = int(val)
                elif "number of indels" in key:
                    parsed["n_indels"] = int(val)
                elif "number of MNPs" in key:
                    parsed["n_mnps"] = int(val)
                elif "number of others" in key:
                    parsed["n_others"] = int(val)
                elif "number of samples" in key:
                    parsed["n_samples"] = int(val)
        elif line.startswith("TSTV\t"):
            parts = line.split("\t")
            if len(parts) >= 5:
                try:
                    parsed["ti_tv_ratio"] = float(parts[4])
                except (ValueError, IndexError):
                    pass
    return parsed


def get_tmp_dir(ctx: Any) -> Path:
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        report_dir = Path("./reports/tmp")
    tmp = Path(report_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp
