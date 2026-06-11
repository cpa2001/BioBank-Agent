"""VCF data loading — convert GATK HaplotypeCaller VCFs to DuckDB-queryable tables.

Registers extracted DataFrames as DuckDB views:
  vcf_variants  — one row per variant (CHROM, POS, REF, ALT, rsID, AF, etc.)
  vcf_genotypes — one row per (sample, variant) pair with GT, AD, DP, GQ
  vcf_samples   — one row per sample with age, sex, phenotype, vcf_path
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def _get_vcf_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env in ("VC_WGS_VCF_DIR", "VC_VIRTUAL_VCF_DIR"):
        val = os.getenv(env, "").strip()
        if val:
            dirs.append(Path(val))
    dirs.extend([
        Path("data/vc_wgs_vcf"),
        Path("data/VirtualCell_WGS_vcf"),
        Path("data/BW_WGS_vcf"),
        Path("input/Files/ResultData/VirtualCell_WGS_vcf"),
        Path("input/Files/ResultData/BW_WGS_vcf"),
        Path("/Files/ResultData/VirtualCell_WGS_vcf"),
        Path("/Files/ResultData/BW_WGS_vcf"),
    ])
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _vcf_stem(path: str | Path) -> str:
    """Return the sample stem used by single-sample genotyper VCF files."""
    name = Path(str(path)).name
    for suffix in (".genotyper.vcf.gz", ".vcf.gz", ".vcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _normalise_manifest_path(path: object, manifest_dir: Path, vcf_dirs: list[Path]) -> str:
    """Resolve manifest filePath values against known local roots."""
    raw = str(path or "").strip()
    if not raw:
        return ""
    candidates: list[Path] = []
    p = Path(raw)
    candidates.append(p)
    if raw.startswith("/Files/"):
        candidates.append(Path(raw))
    if not p.is_absolute():
        candidates.append(manifest_dir / p)
    if raw.startswith("/Files/ResultData/"):
        candidates.append(Path(raw))
    for vcf_dir in vcf_dirs:
        candidates.append(vcf_dir / Path(raw).name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def load_sample_manifest(
    manifest_path: str | Path,
    vcf_dirs: Optional[list[Path]] = None,
) -> pd.DataFrame:
    """Load a WGS sample manifest from .xlsx, .csv, .tsv, or parquet.

    The returned frame follows the VirtualCell schema used by the WGS skills:
    ``sample_id, donor, part, age, sex, is_male, phenotype, phenotype_group,
    vcf_path``. The ``sample_id`` is normalised to match the VCF sample/stem
    (for example ``J1-41Y-F``), while a source ``sampleID`` is preserved as
    ``source_sample_id`` when present.
    """
    manifest = Path(manifest_path).expanduser()
    vcf_dirs = vcf_dirs or _get_vcf_dirs()
    suffix = manifest.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(manifest)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel WGS manifests requires openpyxl. "
                "Install it in the biobank-agent environment or provide CSV/TSV/parquet."
            ) from exc
    elif suffix == ".parquet":
        df = pd.read_parquet(manifest)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(manifest, sep="\t")
    else:
        df = pd.read_csv(manifest)

    rename_map = {
        "Donor": "donor",
        "Part": "part",
        "Age": "age",
        "Sex": "sex",
        "Phenotype": "phenotype",
        "filePath": "vcf_path",
        "FilePath": "vcf_path",
        "filepath": "vcf_path",
        "sampleID": "source_sample_id",
        "SampleID": "source_sample_id",
        "sampleId": "source_sample_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "source_sample_id" not in df.columns and "sample_id" in df.columns:
        df["source_sample_id"] = df["sample_id"]

    if "vcf_path" not in df.columns:
        raise ValueError("WGS sample manifest must include a filePath/vcf_path column.")

    df["vcf_path"] = [
        _normalise_manifest_path(v, manifest.parent, vcf_dirs)
        for v in df["vcf_path"]
    ]
    df["sample_id"] = [_vcf_stem(v) for v in df["vcf_path"]]

    if "donor" not in df.columns:
        df["donor"] = df["sample_id"].astype(str).str.split("-").str[0]
    if "phenotype" not in df.columns:
        df["phenotype"] = ""
    if "phenotype_group" not in df.columns:
        first = df["donor"].astype(str).str[0].str.upper()
        phenotype_group = df["phenotype"].astype(str).str.extract(r"^(Juvenile|Senile|Vitiligo)", expand=False)
        phenotype_group = phenotype_group.map({"Juvenile": "J", "Senile": "S", "Vitiligo": "V"})
        df["phenotype_group"] = phenotype_group.fillna(first)
    if "sex" in df.columns:
        sex_norm = df["sex"].astype(str).str.strip().str.upper().str[0]
        df["sex"] = sex_norm.map({"M": "M", "F": "F"}).fillna(df["sex"])
        df["is_male"] = (sex_norm == "M").astype(int)
    elif "is_male" not in df.columns:
        df["sex"] = ""
        df["is_male"] = 0
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")
    else:
        df["age"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "part" not in df.columns:
        df["part"] = ""

    cols = [
        "sample_id", "donor", "part", "age", "sex", "is_male",
        "phenotype", "phenotype_group", "vcf_path",
    ]
    if "source_sample_id" in df.columns:
        cols.append("source_sample_id")
    out = df[cols].copy()
    return out


def discover_sample_manifest(data_dir: str | Path, raw_dir: str | Path | None = None) -> Path | None:
    """Find a likely WGS sample manifest in configured data roots."""
    for env_name in ("WGS_SAMPLE_INFO", "VC_WGS_SAMPLE_INFO", "VC_SAMPLE_MANIFEST"):
        value = os.getenv(env_name, "").strip()
        if value and Path(value).expanduser().exists():
            return Path(value).expanduser()
    roots = [Path(data_dir)]
    if raw_dir is not None:
        roots.append(Path(raw_dir))
    roots.extend([
        Path("input"),
        Path("input/Files"),
        Path("input/Files/ResultData"),
        Path("/Files/ResultData"),
    ])
    names = (
        "WGS_Sample_info.xlsx",
        "WGS_Sample_info.tsv",
        "WGS_Sample_info.csv",
        "vc_samples.parquet",
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def discover_vcf_files(*dirs: Path) -> list[dict]:
    """Scan directories for .genotyper.vcf.gz files and return path dicts."""
    files: list[dict] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.genotyper.vcf.gz")):
            name = f.stem.replace(".genotyper.vcf", "")
            if name in seen:
                continue
            seen.add(name)
            tbi = f.parent / (f.name + ".tbi")
            files.append({
                "filename": name,
                "vcf_path": str(f),
                "has_index": tbi.exists(),
                "size_mb": round(f.stat().st_size / 1048576, 1),
            })
    return files


def extract_variants_from_vcf(
    vcf_path: str,
    sample_id: str,
    max_variants: int = 0,
    region: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a VCF and return (variants_df, genotypes_df)."""
    try:
        import cyvcf2
    except ImportError:
        raise ImportError("cyvcf2 is required for VCF parsing: pip install cyvcf2")

    vcf = cyvcf2.VCF(vcf_path)
    variants: list[dict] = []
    genotypes: list[dict] = []
    count = 0

    iterator = vcf(region) if region else vcf
    for variant in iterator:
        if max_variants and count >= max_variants:
            break

        alt = ",".join(variant.ALT) if variant.ALT else "."
        vid = f"{variant.CHROM}:{variant.POS}:{variant.REF}:{alt}"

        variants.append({
            "variant_id": vid,
            "chrom": variant.CHROM,
            "pos": variant.POS,
            "ref": variant.REF,
            "alt": alt,
            "rsid": variant.ID or ".",
            "qual": variant.QUAL,
            "filter": variant.FILTER or "PASS",
            "ac": variant.INFO.get("AC"),
            "af": variant.INFO.get("AF"),
            "an": variant.INFO.get("AN"),
            "dp_info": variant.INFO.get("DP"),
            "qd": variant.INFO.get("QD"),
            "fs": variant.INFO.get("FS"),
            "mq": variant.INFO.get("MQ"),
        })

        gt_arr = variant.genotypes[0]
        a1, a2, phased = gt_arr[0], gt_arr[1], gt_arr[2]
        gt_str = f"{a1}|{a2}" if phased else f"{a1}/{a2}"
        gt_num = -1 if a1 < 0 or a2 < 0 else (a1 + a2)

        ad = variant.format("AD")
        ad_ref = int(ad[0][0]) if ad is not None and len(ad[0]) >= 1 else None
        ad_alt = int(ad[0][1]) if ad is not None and len(ad[0]) >= 2 else None
        dp = variant.format("DP")
        dp_val = int(dp[0][0]) if dp is not None else None
        gq = variant.format("GQ")
        gq_val = int(gq[0][0]) if gq is not None else None

        genotypes.append({
            "variant_id": vid,
            "sample_id": sample_id,
            "gt": gt_str,
            "gt_numeric": gt_num,
            "ad_ref": ad_ref,
            "ad_alt": ad_alt,
            "dp": dp_val,
            "gq": gq_val,
        })
        count += 1

    vcf.close()
    return pd.DataFrame(variants), pd.DataFrame(genotypes)


class VCFDataManager:
    """DuckDB-backed VCF data layer for the VirtualCell cohort."""

    def __init__(
        self,
        vcf_dirs: Optional[list[Path]] = None,
        conn: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        self.vcf_dirs = vcf_dirs or _get_vcf_dirs()
        self.conn = conn or duckdb.connect(":memory:")
        self._vcf_files = discover_vcf_files(*self.vcf_dirs)

    def list_vcf_files(self) -> list[dict]:
        return self._vcf_files

    def query_variants(
        self,
        sample_ids: Optional[list[str]] = None,
        region: Optional[str] = None,
        max_variants_per_sample: int = 10_000,
    ) -> pd.DataFrame:
        targets = self._vcf_files
        if sample_ids:
            sid_set = set(sample_ids)
            targets = [f for f in targets if f["filename"] in sid_set]

        all_variants: list[pd.DataFrame] = []
        all_gts: list[pd.DataFrame] = []
        for f in targets:
            try:
                vdf, gdf = extract_variants_from_vcf(
                    f["vcf_path"], f["filename"],
                    max_variants=max_variants_per_sample,
                    region=region,
                )
                all_variants.append(vdf)
                all_gts.append(gdf)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", f["filename"], e)

        if not all_variants:
            return pd.DataFrame()

        variants_union = pd.concat(all_variants).drop_duplicates(subset=["variant_id"])
        gts_union = pd.concat(all_gts)

        self.conn.register("vcf_variants", variants_union)
        self.conn.register("vcf_genotypes", gts_union)
        return variants_union

    def variant_stats(self, region: Optional[str] = None) -> dict:
        _ = self.query_variants(region=region, max_variants_per_sample=50_000)
        stats: dict = {}
        try:
            stats["total_variants"] = self.conn.execute(
                "SELECT COUNT(*) FROM vcf_variants"
            ).fetchone()[0]
            stats["snvs"] = self.conn.execute(
                "SELECT COUNT(*) FROM vcf_variants "
                "WHERE LENGTH(ref)=1 AND LENGTH(alt)=1"
            ).fetchone()[0]
            stats["indels"] = stats["total_variants"] - stats["snvs"]
        except Exception as e:
            stats["error"] = str(e)
        return stats
