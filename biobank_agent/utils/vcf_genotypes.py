"""Genotype matrix construction from merged multi-sample VCFs."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cache_dir() -> Path | None:
    value = (
        os.getenv("WGS_GENOTYPE_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_GENOTYPE_CACHE_DIR", "").strip()
        or os.getenv("WGS_VCF_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_VCF_CACHE_DIR", "").strip()
    )
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_signature(path: str | Path) -> dict:
    p = Path(path)
    try:
        stat = p.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"missing": True}


def _cache_key(operation: str, merged_vcf_path: str | Path, params: dict) -> str:
    payload = {
        "version": 1,
        "operation": operation,
        "input": _path_signature(merged_vcf_path),
        "params": params,
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _load_matrix_cache(path: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        G = data["G"]
        vids = data["vids"].astype(str).tolist()
        samples = data["samples"].astype(str).tolist()
        return G, vids, samples
    except Exception as exc:
        logger.debug("Ignoring unreadable genotype matrix cache %s: %s", path, exc)
        return None


def _write_matrix_cache(path: Path, G: np.ndarray, vids: list[str], samples: list[str]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, G=G, vids=np.asarray(vids, dtype=str), samples=np.asarray(samples, dtype=str))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_counts_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        return pd.read_json(path, orient="records").to_dict(orient="records")
    except Exception as exc:
        logger.debug("Ignoring unreadable allele count cache %s: %s", path, exc)
        return None


def _write_counts_cache(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        pd.DataFrame(rows).to_json(tmp, orient="records")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_genotype_matrix(
    merged_vcf_path: str | Path,
    maf_min: float = 0.05,
    maf_max: float = 0.95,
    max_variants: int = 50000,
    snv_only: bool = True,
    region: Optional[str] = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a sample x variant genotype matrix from a merged VCF.

    Returns (G, variant_ids, sample_order) where G[i,j] is the
    alt allele dosage (0/1/2) for sample i at variant j.
    Missing genotypes are mean-imputed per variant.
    """
    cache_root = _cache_dir()
    cache_path = None
    if cache_root is not None:
        key = _cache_key(
            "build_genotype_matrix",
            merged_vcf_path,
            {
                "maf_min": float(maf_min),
                "maf_max": float(maf_max),
                "max_variants": int(max_variants or 0),
                "snv_only": bool(snv_only),
                "region": region or "",
            },
        )
        cache_path = cache_root / f"genotype_matrix_{key}.npz"
        cached = _load_matrix_cache(cache_path)
        if cached is not None:
            return cached

    import cyvcf2

    vcf = cyvcf2.VCF(str(merged_vcf_path))
    samples = list(vcf.samples)
    n_samples = len(samples)

    rows: list[np.ndarray] = []
    vids: list[str] = []

    iterator = vcf(region) if region else vcf
    for variant in iterator:
        if max_variants and len(vids) >= max_variants:
            break

        if snv_only and not (len(variant.REF) == 1 and len(variant.ALT) == 1 and len(variant.ALT[0]) == 1):
            continue

        if not variant.ALT or variant.ALT[0] == ".":
            continue

        gt_types = variant.gt_types
        valid = gt_types != 3
        n_valid = valid.sum()
        if n_valid < n_samples * 0.5:
            continue

        dosage = gt_types.astype(np.float64)
        dosage[gt_types == 3] = np.nan

        alt_freq = np.nanmean(dosage) / 2.0
        if alt_freq < maf_min or alt_freq > maf_max:
            continue

        rows.append(dosage.copy())
        alt = ",".join(variant.ALT)
        vids.append(f"{variant.CHROM}:{variant.POS}:{variant.REF}:{alt}")

    vcf.close()

    if not rows:
        G_empty = np.empty((n_samples, 0))
        if cache_path is not None:
            _write_matrix_cache(cache_path, G_empty, [], samples)
        return G_empty, [], samples

    G = np.column_stack(rows)

    for j in range(G.shape[1]):
        col = G[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmean(col)

    logger.info("Built genotype matrix: %d samples x %d variants", G.shape[0], G.shape[1])
    if cache_path is not None:
        _write_matrix_cache(cache_path, G, vids, samples)
    return G, vids, samples


def build_allele_counts(
    merged_vcf_path: str | Path,
    samples_case: list[str],
    samples_ctrl: list[str],
    maf_min: float = 0.0,
    region: Optional[str] = None,
    max_variants: int = 0,
) -> list[dict]:
    """Build per-variant allele count tables for case-control association.

    Returns list of dicts with variant info and allele counts.
    """
    cache_root = _cache_dir()
    cache_path = None
    if cache_root is not None:
        key = _cache_key(
            "build_allele_counts",
            merged_vcf_path,
            {
                "samples_case": list(samples_case),
                "samples_ctrl": list(samples_ctrl),
                "maf_min": float(maf_min),
                "region": region or "",
                "max_variants": int(max_variants or 0),
            },
        )
        cache_path = cache_root / f"allele_counts_{key}.json"
        cached = _load_counts_cache(cache_path)
        if cached is not None:
            return cached

    import cyvcf2

    vcf = cyvcf2.VCF(str(merged_vcf_path))
    all_samples = list(vcf.samples)

    case_idx = [all_samples.index(s) for s in samples_case if s in all_samples]
    ctrl_idx = [all_samples.index(s) for s in samples_ctrl if s in all_samples]

    if not case_idx or not ctrl_idx:
        vcf.close()
        if cache_path is not None:
            _write_counts_cache(cache_path, [])
        return []

    results: list[dict] = []
    count = 0
    iterator = vcf(region) if region else vcf

    for variant in iterator:
        if max_variants and count >= max_variants:
            break

        if not variant.ALT or variant.ALT[0] == ".":
            continue
        if not (len(variant.REF) == 1 and len(variant.ALT) == 1 and len(variant.ALT[0]) == 1):
            continue

        gt_types = variant.gt_types

        case_gt = gt_types[case_idx]
        ctrl_gt = gt_types[ctrl_idx]

        case_valid = case_gt[case_gt != 3]
        ctrl_valid = ctrl_gt[ctrl_gt != 3]

        if len(case_valid) < 2 or len(ctrl_valid) < 2:
            continue

        case_alt = int(case_valid.sum())
        case_ref = int(len(case_valid) * 2 - case_alt)
        ctrl_alt = int(ctrl_valid.sum())
        ctrl_ref = int(len(ctrl_valid) * 2 - ctrl_alt)

        total_alt = case_alt + ctrl_alt
        total_ref = case_ref + ctrl_ref
        total = total_alt + total_ref
        if total == 0:
            continue

        maf = min(total_alt, total_ref) / total
        if maf < maf_min:
            continue

        results.append({
            "chrom": variant.CHROM,
            "pos": variant.POS,
            "ref": variant.REF,
            "alt": variant.ALT[0],
            "rsid": variant.ID or ".",
            "case_alt": case_alt,
            "case_ref": case_ref,
            "ctrl_alt": ctrl_alt,
            "ctrl_ref": ctrl_ref,
            "maf": round(maf, 6),
        })
        count += 1

    vcf.close()
    if cache_path is not None:
        _write_counts_cache(cache_path, results)
    return results


def get_sample_vcf_paths(ctx) -> dict[str, str]:
    """Query biomarkers view for sample_id → vcf_path mapping."""
    dm = ctx.dm
    try:
        df = dm.query("SELECT sample_id, vcf_path FROM biomarkers")
        return dict(zip(df["sample_id"], df["vcf_path"]))
    except Exception:
        pass

    from biobank_agent.data.vcf_loader import discover_vcf_files
    from biobank_agent.skills.vcf_query import _get_vcf_dirs

    files = discover_vcf_files(*_get_vcf_dirs(ctx))
    return {f["filename"]: f["vcf_path"] for f in files}
