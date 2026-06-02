"""VirtualCell/BWhair multimodal data skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biobank_agent.data.virtualcell_multimodal import (
    h5ad_sample_summaries,
    link_wgs_to_bwhair,
    load_bwhair_manifest,
    load_default_wgs_manifest,
    virtualcell_inventory,
)
from biobank_agent.registry import skill


def _out_dir(ctx: Any, name: str = "00_Cohort") -> Path:
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        settings = getattr(ctx, "settings", None)
        report_dir = getattr(settings, "reports_dir", Path("./reports")) if settings else Path("./reports")
    out = Path(report_dir) / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_table(ctx: Any, key: str, df: pd.DataFrame) -> None:
    try:
        state = getattr(ctx, "state", None)
        if state is not None and hasattr(state, "custom_data"):
            state.custom_data[key] = df.head(200).to_dict(orient="records")
    except Exception:
        pass


@skill(
    name="virtualcell_data_inventory",
    description=(
        "Inventory the embedded/real VirtualCell BWhair multimodal data: WGS VCF, "
        "Stereo-seq h5ad, scRNA h5ad and scATAC h5ad files, with path readiness."
    ),
    parameters={
        "modalities": {
            "type": "string",
            "description": "Comma-separated modalities to summarize: all,wgs,spatial,scrna,scatac",
            "default": "all",
        },
        "inspect_h5ad": {
            "type": "boolean",
            "description": "Open a small number of h5ad files in backed mode for metadata inspection",
            "default": False,
        },
        "max_h5ad_files": {
            "type": "integer",
            "description": "Maximum h5ad files to inspect when inspect_h5ad=true",
            "default": 3,
        },
    },
    required=[],
)
def virtualcell_data_inventory(
    modalities: str = "all",
    inspect_h5ad: bool = False,
    max_h5ad_files: int = 3,
    *,
    ctx=None,
) -> dict:
    wgs = load_default_wgs_manifest()
    h5ad = load_bwhair_manifest()
    requested = {m.strip().lower() for m in str(modalities or "all").split(",") if m.strip()}
    include_all = not requested or "all" in requested
    if not include_all:
        if "wgs" not in requested:
            wgs = wgs.iloc[0:0].copy()
        h5ad = h5ad[h5ad["modality"].isin(requested)].copy()

    inventory = virtualcell_inventory()
    out = _out_dir(ctx, "00_Cohort")
    wgs_path = out / "virtualcell_wgs_manifest.tsv"
    h5ad_path = out / "virtualcell_h5ad_manifest.tsv"
    inv_path = out / "virtualcell_inventory.json"
    wgs.to_csv(wgs_path, sep="\t", index=False)
    h5ad.to_csv(h5ad_path, sep="\t", index=False)

    inspected = []
    if inspect_h5ad:
        inspected = h5ad_sample_summaries(max_files=max(0, int(max_h5ad_files or 0)), include_obs_summary=False)

    payload = {
        "modalities_requested": sorted(requested) if requested else ["all"],
        "inventory": inventory,
        "wgs_manifest": str(wgs_path),
        "h5ad_manifest": str(h5ad_path),
        "h5ad_inspection": inspected,
    }
    _write_json(inv_path, payload)
    _record_table(ctx, "virtualcell_wgs_manifest", wgs)
    _record_table(ctx, "virtualcell_h5ad_manifest", h5ad)

    return {
        **payload,
        "inventory_json": str(inv_path),
        "n_wgs_samples": int(len(wgs)),
        "n_h5ad_files": int(len(h5ad)),
        "n_h5ad_existing": int(h5ad["file_exists"].sum()) if "file_exists" in h5ad else 0,
        "h5ad_modalities": h5ad["modality"].value_counts().to_dict() if not h5ad.empty else {},
        "wgs_phenotypes": wgs["phenotype"].value_counts().to_dict() if not wgs.empty else {},
    }


@skill(
    name="virtualcell_multimodal_link",
    description=(
        "Link VirtualCell WGS donors to BWhair h5ad modalities using Donor as the "
        "cross-modal key, reporting missing/extra donors and hair-state coverage."
    ),
    parameters={
        "include_senile": {
            "type": "boolean",
            "description": "Include Senile_White WGS donors in linkage summary",
            "default": True,
        },
    },
    required=[],
)
def virtualcell_multimodal_link(include_senile: bool = True, *, ctx=None) -> dict:
    link = link_wgs_to_bwhair()
    if not include_senile and "phenotype" in link.columns:
        link = link[~link["phenotype"].astype(str).str.contains("Senile_White", na=False)]
    out = _out_dir(ctx, "00_Cohort")
    path = out / "virtualcell_multimodal_linkage.tsv"
    link.to_csv(path, sep="\t", index=False)
    _record_table(ctx, "virtualcell_multimodal_linkage", link)

    with_wgs = link[link["has_wgs"] == True]  # noqa: E712
    with_h5ad = link[link["n_h5ad_files"] > 0]
    wgs_without_h5ad = link[(link["has_wgs"] == True) & (link["n_h5ad_files"] == 0)]  # noqa: E712
    h5ad_without_wgs = link[(link["has_wgs"] == False) & (link["n_h5ad_files"] > 0)]  # noqa: E712

    return {
        "n_donors_total": int(len(link)),
        "n_wgs_donors": int(len(with_wgs)),
        "n_h5ad_donors": int(len(with_h5ad)),
        "n_linked_donors": int(len(link[(link["has_wgs"] == True) & (link["n_h5ad_files"] > 0)])),  # noqa: E712
        "wgs_without_h5ad": wgs_without_h5ad["donor"].tolist(),
        "h5ad_without_wgs": h5ad_without_wgs["donor"].tolist(),
        "linkage_table": str(path),
        "preview": link.head(20).to_dict(orient="records"),
    }


@skill(
    name="h5ad_sample_summary",
    description=(
        "Inspect BWhair h5ad files safely in backed read-only mode and summarize "
        "obs/var metadata without loading the full matrix into memory."
    ),
    parameters={
        "modality": {
            "type": "string",
            "description": "all, spatial, scrna or scatac",
            "default": "all",
        },
        "sample_query": {
            "type": "string",
            "description": "Optional sample, donor or filename substring filter",
            "default": "",
        },
        "max_files": {
            "type": "integer",
            "description": "Maximum files to inspect; 0 means all matching files",
            "default": 5,
        },
        "include_obs_summary": {
            "type": "boolean",
            "description": "Summarize low-cardinality obs columns where available",
            "default": True,
        },
    },
    required=[],
)
def h5ad_sample_summary(
    modality: str = "all",
    sample_query: str = "",
    max_files: int = 5,
    include_obs_summary: bool = True,
    *,
    ctx=None,
) -> dict:
    summaries = h5ad_sample_summaries(
        modality=modality,
        sample_query=sample_query,
        max_files=int(max_files or 0),
        include_obs_summary=include_obs_summary,
    )
    out = _out_dir(ctx, "00_Cohort")
    path = out / "h5ad_sample_summary.json"
    _write_json(path, summaries)
    status_counts: dict[str, int] = {}
    for item in summaries:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "n_files_summarized": len(summaries),
        "status_counts": status_counts,
        "summary_json": str(path),
        "summaries": summaries[:20],
    }


@skill(
    name="spatial_hair_summary",
    description=(
        "Summarize BWhair Stereo-seq samples by donor and hair state (B/W/WB/G/GB), "
        "including manifest cell counts and optional file-readiness checks."
    ),
    parameters={
        "group_by": {
            "type": "string",
            "description": "Column used for grouping: hair_state, donor, phenotype, or sex",
            "default": "hair_state",
        },
    },
    required=[],
)
def spatial_hair_summary(group_by: str = "hair_state", *, ctx=None) -> dict:
    h5ad = load_bwhair_manifest()
    spatial = h5ad[h5ad["modality"] == "spatial"].copy()
    wgs = load_default_wgs_manifest()[["donor", "phenotype", "phenotype_group"]]
    spatial = spatial.merge(wgs.drop_duplicates("donor"), on="donor", how="left")
    if group_by not in spatial.columns:
        return {"error": f"Column {group_by!r} not found.", "available_columns": list(spatial.columns)}
    summary = spatial.groupby(group_by, dropna=False).agg(
        n_files=("filename", "count"),
        n_donors=("donor", "nunique"),
        cells=("cells", "sum"),
        existing_files=("file_exists", "sum"),
    ).reset_index()
    out = _out_dir(ctx, "00_Cohort")
    summary_path = out / f"spatial_hair_summary_by_{group_by}.tsv"
    sample_path = out / "spatial_hair_manifest_linked.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    spatial.to_csv(sample_path, sep="\t", index=False)
    _record_table(ctx, "spatial_hair_summary", summary)
    return {
        "n_spatial_files": int(len(spatial)),
        "n_spatial_donors": int(spatial["donor"].nunique()),
        "group_by": group_by,
        "summary_table": str(summary_path),
        "linked_manifest": str(sample_path),
        "groups": summary.to_dict(orient="records"),
    }


@skill(
    name="singlecell_modality_summary",
    description=(
        "Summarize BWhair scRNA-seq and scATAC-seq h5ad modalities, including "
        "manifest-level cell counts and backed-mode obs metadata when readable."
    ),
    parameters={
        "modality": {
            "type": "string",
            "description": "scrna, scatac, or all",
            "default": "all",
        },
        "inspect_h5ad": {
            "type": "boolean",
            "description": "Open matching h5ad files in backed mode to read obs/var metadata",
            "default": True,
        },
    },
    required=[],
)
def singlecell_modality_summary(modality: str = "all", inspect_h5ad: bool = True, *, ctx=None) -> dict:
    h5ad = load_bwhair_manifest()
    single = h5ad[h5ad["modality"].isin(["scrna", "scatac"])].copy()
    modality_key = str(modality or "all").lower().strip()
    if modality_key != "all":
        single = single[single["modality"] == modality_key]
    summary = single.groupby("modality", dropna=False).agg(
        n_files=("filename", "count"),
        cells=("cells", "sum"),
        size_mb=("size_mb", "sum"),
        existing_files=("file_exists", "sum"),
    ).reset_index()
    inspections = []
    if inspect_h5ad:
        inspections = h5ad_sample_summaries(
            modality=modality_key if modality_key != "all" else "all",
            sample_query="BWhair_",
            max_files=2,
            include_obs_summary=False,
        )
    out = _out_dir(ctx, "00_Cohort")
    table_path = out / "singlecell_modality_summary.tsv"
    json_path = out / "singlecell_h5ad_inspection.json"
    summary.to_csv(table_path, sep="\t", index=False)
    _write_json(json_path, inspections)
    _record_table(ctx, "singlecell_modality_summary", summary)
    return {
        "n_files": int(len(single)),
        "modalities": summary.to_dict(orient="records"),
        "summary_table": str(table_path),
        "inspection_json": str(json_path),
        "inspection_status_counts": {
            status: sum(1 for item in inspections if item.get("status") == status)
            for status in sorted({str(item.get("status")) for item in inspections})
        } if inspections else {},
        "inspection_policy": "bounded_backed_metadata; large scRNA/scATAC matrices are not fully loaded in default CLI/harness runs",
        "inspections": inspections[:10],
    }
