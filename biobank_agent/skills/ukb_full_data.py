"""UKB full-data inventory and materialization skills."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from biobank_agent.data.parquet_builder import (
    FULL_UKB_SOURCES,
    build_full_ukb_feature_store,
    full_ukb_inventory,
)
from biobank_agent.registry import skill
from biobank_agent.skills.data_query import field_search


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _report_dir(ctx: Any, output_dir: str = "") -> Path:
    if output_dir:
        return Path(output_dir).expanduser()
    return Path(getattr(ctx, "report_dir", getattr(ctx.settings, "reports_dir", "./reports")))


def _write_json(report_dir: Path, prefix: str, payload: dict[str, Any]) -> str:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


@skill(
    name="ukb_data_inventory",
    description=(
        "Inspect current UKB data coverage across Milton parquet, raw UKB CSVs, "
        "and the optional full UKB feature-store manifest. Reads headers and "
        "aggregate counts only; does not sample participant-level data."
    ),
    parameters={
        "sources": {
            "type": "string",
            "description": "Comma-separated full UKB sources to inspect; empty means all known sources",
            "default": "",
        },
        "output_dir": {
            "type": "string",
            "description": "Optional directory for JSON inventory artifact; defaults to report_dir",
            "default": "",
        },
    },
    required=[],
)
def ukb_data_inventory(sources: str = "", output_dir: str = "", *, ctx=None) -> dict:
    if ctx is None:
        return {"error": "No agent context is available.", "status": "ERROR"}

    dm = getattr(ctx, "dm", None)
    settings = ctx.settings
    source_list = _split_csv(sources) or None
    inventory = full_ukb_inventory(settings.raw_dir, sources=source_list)
    feature_store = settings.full_ukb_feature_store
    manifest_path = feature_store / "manifest.json"
    manifest_summary = {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
    }
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_summary.update({
                "status": manifest.get("status", ""),
                "n_chunks": manifest.get("n_chunks", 0),
                "n_sources": len(manifest.get("sources", {}) or {}),
            })
        except Exception as exc:
            manifest_summary["error"] = str(exc)

    payload = {
        "status": "READY" if inventory.get("status") == "READY" else "PARTIAL",
        "data_dir": str(settings.data_dir),
        "raw_dir": str(settings.raw_dir),
        "full_ukb_feature_store": str(feature_store),
        "milton_parquet": {
            "biomarkers": str(settings.biomarker_parquet),
            "diagnoses": str(settings.diagnoses_parquet),
            "deaths": str(settings.deaths_parquet),
            "categories": str(settings.category_parquet_dir),
        },
        "full_csv_inventory": inventory,
        "feature_store_manifest": manifest_summary,
        "n_subjects": None,
        "n_registered_biomarker_columns": None,
        "n_registered_field_prefixes": None,
    }
    if dm is not None:
        try:
            payload["n_subjects"] = int(dm.count_subjects())
        except Exception as exc:
            payload.setdefault("warnings", []).append(f"count_subjects failed: {exc}")
        try:
            payload["n_registered_biomarker_columns"] = len(dm.list_parquet_columns())
            payload["n_registered_field_prefixes"] = len(dm._get_parquet_fields())
        except Exception as exc:
            payload.setdefault("warnings", []).append(f"field introspection failed: {exc}")

    payload["artifact_json"] = _write_json(_report_dir(ctx, output_dir), "ukb_data_inventory", payload)
    return payload


@skill(
    name="ukb_field_resolve",
    description=(
        "Resolve a UKB research query or explicit field IDs to catalogue fields "
        "and concrete data sources, including raw main CSV fields and full "
        "feature-store chunks when available."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Biomedical query such as 'diabetes medication blood pressure HbA1c'",
            "default": "",
        },
        "field_ids": {
            "type": "string",
            "description": "Optional comma-separated field IDs to resolve exactly",
            "default": "",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum catalogue hits for query search",
            "default": 30,
        },
    },
    required=[],
)
def ukb_field_resolve(query: str = "", field_ids: str = "", limit: int = 30, *, ctx=None) -> dict:
    if ctx is None:
        return {"error": "No agent context is available.", "status": "ERROR"}
    dm = getattr(ctx, "dm", None)
    catalog = getattr(ctx, "catalog", None)
    resolved: dict[str, dict[str, Any]] = {}

    for fid in _split_csv(field_ids):
        info = catalog.field_info(fid) if catalog is not None else None
        source = dm.field_source(fid) if dm is not None else "unknown"
        resolved[fid] = {
            "field_id": fid,
            "title": (info or {}).get("title", ""),
            "category": catalog.category_name((info or {}).get("category_id", "")) if catalog is not None and info else "",
            "value_type": (info or {}).get("value_type", ""),
            "units": (info or {}).get("units", ""),
            "data_source": source,
            "needs_materialization": source.startswith("main_") or source in {"unknown"},
        }

    search_result = {}
    if query:
        search_result = field_search(query=query, limit=limit, ctx=ctx)
        for row in search_result.get("results", []) or []:
            fid = str(row.get("field_id", ""))
            if fid and fid not in resolved:
                source = dm.field_source(fid) if dm is not None else row.get("data_source", "unknown")
                item = dict(row)
                item["data_source"] = source
                item["needs_materialization"] = source.startswith("main_") or source == "unknown"
                resolved[fid] = item

    return {
        "status": "READY" if resolved else "PARTIAL",
        "query": query,
        "field_ids": list(resolved),
        "fields": list(resolved.values()),
        "n_fields": len(resolved),
        "materialization_candidates": [
            fid for fid, row in resolved.items()
            if row.get("needs_materialization")
        ],
        "search": {
            "matched_queries": search_result.get("matched_queries", []),
            "searched_queries": search_result.get("searched_queries", []),
            "warnings": search_result.get("warnings", []),
        } if search_result else {},
    }


@skill(
    name="ukb_materialize_fields",
    description=(
        "Materialize selected full-UKB raw CSV fields into the partitioned "
        "feature-store parquet cache. Use this before modelling when important "
        "fields exist only in the UKB main CSVs."
    ),
    parameters={
        "field_ids": {
            "type": "string",
            "description": "Comma-separated UKB field IDs to materialize",
        },
        "sources": {
            "type": "string",
            "description": "Optional comma-separated source keys; empty means all full UKB sources",
            "default": "",
        },
        "output_dir": {
            "type": "string",
            "description": "Feature-store output directory; defaults to settings.full_ukb_feature_store",
            "default": "",
        },
        "chunk_cols": {
            "type": "integer",
            "description": "Maximum feature columns per parquet chunk",
            "default": 250,
        },
        "dry_run": {
            "type": "boolean",
            "description": "If true, only report what would be written",
            "default": False,
        },
    },
    required=["field_ids"],
)
def ukb_materialize_fields(
    field_ids: str,
    sources: str = "",
    output_dir: str = "",
    chunk_cols: int = 250,
    dry_run: bool = False,
    *,
    ctx=None,
) -> dict:
    if ctx is None:
        return {"error": "No agent context is available.", "status": "ERROR"}
    fields = _split_csv(field_ids)
    if not fields:
        return {"error": "field_ids is required; refusing to materialize every UKB field from an agent step.", "status": "ERROR"}
    source_list = _split_csv(sources) or None
    target = Path(output_dir).expanduser() if output_dir else ctx.settings.full_ukb_feature_store

    def progress(source: str, msg: str) -> None:
        callback = getattr(ctx, "emit_progress", None)
        if callable(callback):
            callback("ukb-materialize", f"{source}: {msg}", {"source": source, "fields": fields})

    result = build_full_ukb_feature_store(
        ctx.settings.raw_dir,
        target,
        sources=source_list,
        field_ids=fields,
        chunk_cols=chunk_cols,
        resume=True,
        dry_run=dry_run,
        callback=progress,
    )
    if not dry_run and getattr(ctx, "dm", None) is not None:
        try:
            ctx.settings.full_ukb_feature_store_dir = str(target)
            ctx.dm.refresh_parquet_views()
        except Exception as exc:
            result.setdefault("warnings", []).append(f"Could not refresh DataManager feature-store views: {exc}")
    if dry_run:
        result["status"] = "READY" if result.get("status") == "READY" else "PARTIAL"
    else:
        result["status"] = "READY" if result.get("n_chunks", 0) else "PARTIAL"
    result["requested_field_ids"] = fields
    result["known_source_keys"] = sorted(FULL_UKB_SOURCES)
    return result
