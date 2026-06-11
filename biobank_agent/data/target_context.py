"""Biobank target annotation and enrichment primitives.

Design source:
    - Open Targets, UniProt, GTEx, ClinicalTrials.gov, CELLxGENE and GSEApy.
    - Concept: external annotations explain biobank target lists but never
      override genetic or epidemiological evidence.
    - Reference doc: docs/related_works/TARGET_ANNOTATION_ENRICHMENT.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_ANNOTATION_SOURCES = ["opentargets", "uniprot", "gtex", "clinicaltrials", "cellxgene"]
ANNOTATION_SOURCE_INFO = {
    "opentargets": {
        "title": "Open Targets Platform GraphQL API",
        "url": "https://platform-docs.opentargets.org/data-access/graphql-api",
    },
    "uniprot": {
        "title": "UniProt REST API",
        "url": "https://www.uniprot.org/help/api",
    },
    "gtex": {
        "title": "GTEx Portal API v2",
        "url": "https://gtexportal.org/api/v2/redoc",
    },
    "clinicaltrials": {
        "title": "ClinicalTrials.gov API v2",
        "url": "https://clinicaltrials.gov/data-about-studies/learn-about-api",
    },
    "cellxgene": {
        "title": "CZ CELLxGENE Census Python API",
        "url": "https://chanzuckerberg.github.io/cellxgene-census/python-api.html",
    },
}


def slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def normalize_gene(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, list) else [parsed]
        return [part.strip() for part in re.split(r"[,;\n]+", stripped) if part.strip()]
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def normalize_targets(targets: Any) -> list[dict[str, Any]]:
    """Normalize a target payload into dictionaries with a gene key."""

    out: list[dict[str, Any]] = []
    for item in _as_list(targets):
        if isinstance(item, str):
            row = {"gene": item}
        elif isinstance(item, dict):
            row = dict(item)
        else:
            continue
        gene = normalize_gene(row.get("gene") or row.get("approvedSymbol") or row.get("symbol"))
        if not gene:
            continue
        row["gene"] = gene
        if row.get("ensembl_id"):
            row["ensembl_id"] = str(row["ensembl_id"]).split(".")[0]
        elif row.get("ensemblId"):
            row["ensembl_id"] = str(row["ensemblId"]).split(".")[0]
        if row.get("uniprot_id"):
            row["uniprot_id"] = str(row["uniprot_id"])
        return_dict = {str(k): v for k, v in row.items()}
        out.append(return_dict)
    return out


def normalize_sources(sources: Any) -> list[str]:
    requested = [str(s).strip().lower() for s in _as_list(sources)]
    if not requested:
        requested = list(DEFAULT_ANNOTATION_SOURCES)
    aliases = {"open_targets": "opentargets", "clinical_trials": "clinicaltrials", "cellx_gene": "cellxgene"}
    out: list[str] = []
    for source in requested:
        source = aliases.get(source, source)
        if source in ANNOTATION_SOURCE_INFO and source not in out:
            out.append(source)
    return out


def invalid_sources(sources: Any) -> list[str]:
    requested = [str(s).strip().lower() for s in _as_list(sources)]
    if not requested:
        return []
    aliases = {"open_targets": "opentargets", "clinical_trials": "clinicaltrials", "cellx_gene": "cellxgene"}
    invalid = []
    for source in requested:
        source = aliases.get(source, source)
        if source and source not in ANNOTATION_SOURCE_INFO and source not in invalid:
            invalid.append(source)
    return invalid


def _safe_positive_int(value: Any, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    out = max(minimum, out)
    if maximum is not None:
        out = min(out, maximum)
    return out


class AnnotationCache:
    """Small JSON cache for source-level API responses."""

    def __init__(self, cache_dir: str | Path | None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, cache_key: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{slug(source)}_{digest}.json"

    def get(self, source: str, cache_key: str) -> dict[str, Any] | None:
        path = self._path(source, cache_key)
        if not path or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, source: str, cache_key: str, payload: dict[str, Any]) -> None:
        path = self._path(source, cache_key)
        if not path:
            return
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


class TargetAnnotationClient:
    """HTTP client wrapper with cache-first and cache-only source modes."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None,
        source_mode: str = "cache_first",
        timeout_s: float = 20.0,
    ) -> None:
        mode = str(source_mode or "cache_first").lower()
        self.source_mode = mode if mode in {"cache_first", "cache_only", "refresh"} else "cache_first"
        self.timeout_s = float(timeout_s or 20.0)
        self.cache = AnnotationCache(cache_dir)

    def request_json(
        self,
        source: str,
        url: str,
        *,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_key = json.dumps(
            {"method": method.upper(), "url": url, "params": params or {}, "json": json_body or {}},
            sort_keys=True,
            default=str,
        )
        cached = self.cache.get(source, cache_key)
        if cached and self.source_mode == "cache_first":
            return {**cached, "cache_hit": True}
        if self.source_mode == "cache_only":
            if cached:
                return {**cached, "cache_hit": True}
            return _source_result(
                source,
                "SKIPPED",
                error="Cache miss in cache_only mode.",
                url=url,
                query={"params": params or {}, "json": json_body or {}},
                cache_hit=False,
            )

        try:
            headers = {
                "Accept": "application/json",
                "User-Agent": "BiobankAgent/2.3 target-context",
            }
            with httpx.Client(timeout=self.timeout_s, follow_redirects=True, headers=headers) as client:
                response = client.request(method.upper(), url, params=params, json=json_body)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            if cached:
                return {
                    **cached,
                    "status": "PARTIAL",
                    "cache_hit": True,
                    "error": f"Live refresh failed; using cached response: {exc}",
                }
            return _source_result(
                source,
                "ERROR",
                error=str(exc),
                url=url,
                query={"params": params or {}, "json": json_body or {}},
                cache_hit=False,
            )

        payload = _source_result(
            source,
            "PASS",
            data=data,
            url=url,
            query={"params": params or {}, "json": json_body or {}},
            cache_hit=False,
        )
        self.cache.set(source, cache_key, payload)
        return payload


def build_target_annotation_context(
    *,
    targets: Any,
    phenotype: str = "",
    disease_id: str = "",
    sources: Any = None,
    source_mode: str = "cache_first",
    cache_dir: str | Path | None = None,
    timeout_s: float = 20.0,
    cellxgene_snapshot_path: str | Path | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Annotate biobank target genes with external translational context."""

    target_rows = normalize_targets(targets)
    requested_sources = normalize_sources(sources)
    invalid_requested = invalid_sources(sources)
    top_n_value = _safe_positive_int(top_n, 5, minimum=1, maximum=100)
    if not target_rows:
        return {
            "skill": "target_annotation_context",
            "status": "NEEDS_INPUT",
            "message": "Provide at least one target gene.",
            "targets": [],
            "sources_requested": requested_sources,
            "sources": _source_metadata(requested_sources),
        }
    if not requested_sources:
        return {
            "skill": "target_annotation_context",
            "status": "INVALID_INPUT",
            "message": "No supported annotation sources were requested.",
            "targets": target_rows,
            "sources_requested": [],
            "invalid_sources": invalid_requested,
            "supported_sources": list(ANNOTATION_SOURCE_INFO),
            "warnings": [f"Unsupported annotation source ignored: {source}" for source in invalid_requested],
            "sources": [],
        }

    client = TargetAnnotationClient(cache_dir=cache_dir, source_mode=source_mode, timeout_s=timeout_s)
    annotated: list[dict[str, Any]] = []
    all_warnings: list[str] = [f"Unsupported annotation source ignored: {source}" for source in invalid_requested]
    source_status_counts: dict[str, dict[str, int]] = {}

    for row in target_rows[:top_n_value]:
        context = _empty_target_context(row, phenotype=phenotype)
        for source in requested_sources:
            source_payload = _call_annotation_source(
                source,
                client,
                context,
                disease_id=disease_id,
                phenotype=phenotype,
                top_n=top_n_value,
                cellxgene_snapshot_path=cellxgene_snapshot_path,
            )

            status = str(source_payload.get("status", "UNKNOWN")).upper()
            source_status_counts.setdefault(source, {}).setdefault(status, 0)
            source_status_counts[source][status] += 1
            context.setdefault("source_results", []).append(_compact_source_result(source_payload))
            context.setdefault("source_status", {})[source] = status
            if source_payload.get("error"):
                warning = f"{source}: {source_payload['error']}"
                context.setdefault("warnings", []).append(warning)
                all_warnings.append(f"{context['gene']}: {warning}")
        annotated.append(context)

    status = _annotation_overall_status(annotated)
    return {
        "skill": "target_annotation_context",
        "status": status,
        "phenotype": phenotype,
        "disease_id": disease_id,
        "source_mode": source_mode,
        "sources_requested": requested_sources,
        "invalid_sources": invalid_requested,
        "source_status_counts": source_status_counts,
        "n_targets": len(annotated),
        "targets": annotated,
        "warnings": all_warnings[:50],
        "design_principle": "genetic_evidence_drives_ranking_annotations_are_context",
        "caveats": [
            "External annotations are translational context and do not change genetic-evidence ranking.",
            "Trial and tractability records are not treatment recommendations.",
            "Tissue and cell-type expression context is aggregate reference evidence, not participant-level inference.",
        ],
        "sources": _source_metadata(requested_sources),
    }


def write_target_annotation_report(result: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    out_dir = Path(report_dir) / "target_annotation_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = slug(result.get("phenotype") or "targets")
    md_path = out_dir / f"{name}_annotations.md"
    json_path = out_dir / f"{name}_annotations.json"

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        f"# Target Annotation Context: {result.get('phenotype') or 'targets'}",
        "",
        f"**Status:** {result.get('status', '')}",
        f"**Source mode:** {result.get('source_mode', '')}",
        f"**Targets:** {result.get('n_targets', 0)}",
        "",
        "| Gene | Ensembl | UniProt | Trials | Top tissues | Source status |",
        "|------|---------|---------|-------:|-------------|---------------|",
    ]
    for target in result.get("targets", []) or []:
        tissues = ", ".join(t.get("tissue", "") for t in (target.get("tissue_expression") or [])[:3])
        status = ", ".join(f"{k}:{v}" for k, v in sorted((target.get("source_status") or {}).items()))
        lines.append(
            "| {gene} | {ens} | {up} | {trials} | {tissues} | {status} |".format(
                gene=target.get("gene", ""),
                ens=target.get("ensembl_id", ""),
                up=target.get("uniprot_id", ""),
                trials=len(target.get("clinical_trials") or []),
                tissues=tissues,
                status=status,
            )
        )
    lines.extend([
        "",
        "## Caveats",
        "",
    ])
    for caveat in result.get("caveats", []) or []:
        lines.append(f"- {caveat}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}


def record_target_annotations_to_action_graph(memory: Any, result: dict[str, Any]) -> None:
    if memory is None or not hasattr(memory, "upsert_node"):
        return
    phenotype = str(result.get("phenotype") or "targets")
    set_id = f"target_annotation_context:{slug(phenotype)}"
    memory.upsert_node(
        "target_annotation_context_set",
        set_id,
        payload={
            "phenotype": result.get("phenotype"),
            "status": result.get("status"),
            "sources_requested": result.get("sources_requested", []),
            "n_targets": result.get("n_targets", 0),
        },
        score=0.8 if result.get("status") == "PASS" else 0.4,
    )
    if result.get("phenotype"):
        memory.upsert_node("phenotype", slug(phenotype), payload={"label": phenotype}, score=0.7)
        memory.link_nodes(
            "target_annotation_context_set",
            set_id,
            "phenotype",
            slug(phenotype),
            relation="annotates_targets_for",
            weight=0.7,
        )

    for target in result.get("targets", [])[:100]:
        gene = normalize_gene(target.get("gene"))
        if not gene:
            continue
        node_id = f"{set_id}:{gene}"
        memory.upsert_node("gene", gene, payload={"gene": gene, "ensembl_id": target.get("ensembl_id")}, score=0.7)
        memory.upsert_node("target_annotation_context", node_id, payload=target, score=0.6)
        memory.link_nodes(
            "target_annotation_context_set",
            set_id,
            "target_annotation_context",
            node_id,
            relation="contains",
            weight=0.7,
        )
        memory.link_nodes(
            "target_annotation_context",
            node_id,
            "gene",
            gene,
            relation="annotates_gene",
            weight=0.8,
        )
        if result.get("phenotype"):
            hypothesis_id = f"genetic_target_hypothesis:{slug(phenotype)}:{gene}"
            memory.link_nodes(
                "therapeutic_target_hypothesis",
                hypothesis_id,
                "target_annotation_context",
                node_id,
                relation="has_translational_context",
                weight=0.5,
                evidence={"source_mode": result.get("source_mode")},
            )
        for source in target.get("source_results", [])[:10]:
            evidence_id = f"{node_id}:{source.get('source', 'source')}"
            memory.upsert_node("external_annotation_source", evidence_id, payload=source, score=0.4)
            memory.link_nodes(
                "target_annotation_context",
                node_id,
                "external_annotation_source",
                evidence_id,
                relation="supported_by_external_source",
                weight=0.4,
                evidence={"status": source.get("status")},
            )


def build_target_enrichment(
    *,
    gene_list: Any = None,
    ranked_genes: Any = None,
    phenotype: str = "",
    gene_sets_path: str | Path | None = None,
    method: str = "ora",
    universe_genes: Any = None,
    fdr_alpha: float = 0.05,
    top_n: int = 20,
) -> dict[str, Any]:
    """Run local gene-set enrichment for a biobank target list."""

    genes, ranks = _normalize_gene_inputs(gene_list=gene_list, ranked_genes=ranked_genes)
    top_n_value = _safe_positive_int(top_n, 20, minimum=1, maximum=500)
    try:
        fdr_alpha_value = float(fdr_alpha)
    except (TypeError, ValueError):
        fdr_alpha_value = 0.05
    if not math.isfinite(fdr_alpha_value) or fdr_alpha_value <= 0 or fdr_alpha_value > 1:
        fdr_alpha_value = 0.05
    if not genes:
        return {
            "skill": "target_enrichment",
            "status": "NEEDS_INPUT",
            "message": "Provide gene_list or ranked_genes.",
            "phenotype": phenotype,
            "terms": [],
            "warnings": [],
        }

    requested_method = str(method or "ora").lower()
    warnings: list[str] = []
    gseapy_result = None
    if requested_method in {"prerank", "gseapy", "enrichr"}:
        gseapy_result = _try_gseapy_enrichment(
            ranked_genes=ranks,
            genes=genes,
            gene_sets_path=gene_sets_path,
            method=requested_method,
            top_n=top_n_value,
            warnings=warnings,
        )
        if gseapy_result is not None:
            return {
                "skill": "target_enrichment",
                **gseapy_result,
                "phenotype": phenotype,
                "input_genes": genes,
                "n_input_genes": len(genes),
                "fdr_alpha": fdr_alpha_value,
                "warnings": warnings,
                "caveats": _enrichment_caveats(),
                "sources": [{"title": "GSEApy", "url": "https://gseapy.readthedocs.io/en/latest/introduction.html"}],
            }
        warnings.append("gseapy was unavailable or failed; using local ORA when a GMT file is supplied.")

    if not gene_sets_path:
        return {
            "skill": "target_enrichment",
            "status": "NEEDS_INPUT",
            "message": "Provide a local GMT gene_sets_path for offline enrichment.",
            "phenotype": phenotype,
            "input_genes": genes,
            "n_input_genes": len(genes),
            "terms": [],
            "warnings": warnings,
            "caveats": _enrichment_caveats(),
        }

    try:
        gene_sets = load_gene_sets(gene_sets_path)
    except OSError as exc:
        return {
            "skill": "target_enrichment",
            "status": "INVALID_INPUT",
            "message": f"Could not read local GMT gene_sets_path: {exc}",
            "phenotype": phenotype,
            "input_genes": genes,
            "n_input_genes": len(genes),
            "terms": [],
            "warnings": warnings,
            "caveats": _enrichment_caveats(),
        }
    if not gene_sets:
        return {
            "skill": "target_enrichment",
            "status": "INVALID_INPUT",
            "message": "No usable gene sets were found in the supplied GMT file.",
            "phenotype": phenotype,
            "input_genes": genes,
            "n_input_genes": len(genes),
            "terms": [],
            "warnings": warnings,
            "caveats": _enrichment_caveats(),
        }

    universe = [normalize_gene(g) for g in _as_list(universe_genes) if normalize_gene(g)]
    explicit_universe = bool(universe)
    if not explicit_universe:
        universe = sorted({gene for genes_in_set in gene_sets.values() for gene in genes_in_set})
        warnings.append("No explicit universe_genes supplied; using the union of GMT genes as the ORA universe.")
    dropped = sorted(set(genes) - set(universe))
    if dropped:
        warnings.append(
            f"{len(dropped)} input gene(s) were outside the enrichment universe and excluded: "
            + ", ".join(dropped[:10])
        )
    terms = _run_local_ora(genes, gene_sets, universe, fdr_alpha=fdr_alpha_value, top_n=top_n_value)
    status = "PASS" if terms else "NO_ENRICHMENT"
    return {
        "skill": "target_enrichment",
        "status": status,
        "phenotype": phenotype,
        "method": "local_ora",
        "gene_sets_path": str(Path(gene_sets_path).expanduser()),
        "explicit_universe": explicit_universe,
        "n_input_genes": len(genes),
        "n_tested_genes": len(set(genes) & set(universe)),
        "dropped_genes": dropped[:50],
        "n_gene_sets": len(gene_sets),
        "n_universe_genes": len(universe),
        "terms": terms,
        "warnings": warnings,
        "fdr_alpha": fdr_alpha_value,
        "caveats": _enrichment_caveats(),
        "sources": [{"title": "Local GMT gene-set file", "url": str(gene_sets_path)}],
    }


def load_gene_sets(path: str | Path) -> dict[str, set[str]]:
    """Load a GMT file as term -> uppercase gene set."""

    gene_sets: dict[str, set[str]] = {}
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = {normalize_gene(g) for g in parts[2:] if normalize_gene(g)}
            if term and genes:
                gene_sets[term] = genes
    return gene_sets


def write_target_enrichment_report(result: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    out_dir = Path(report_dir) / "target_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = slug(result.get("phenotype") or "targets")
    md_path = out_dir / f"{name}_enrichment.md"
    csv_path = out_dir / f"{name}_enrichment.csv"

    terms = result.get("terms", []) or []
    fieldnames = ["rank", "term", "p_value", "q_value", "overlap", "overlap_genes"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in terms:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    lines = [
        f"# Target Enrichment: {result.get('phenotype') or 'targets'}",
        "",
        f"**Status:** {result.get('status', '')}",
        f"**Method:** {result.get('method', '')}",
        f"**Input genes:** {result.get('n_input_genes', 0)}",
        f"**Gene sets:** {result.get('n_gene_sets', 0)}",
        "",
        "| Rank | Term | Overlap | P-value | Q-value | Genes |",
        "|------|------|--------:|--------:|--------:|-------|",
    ]
    for row in terms[:50]:
        lines.append(
            "| {rank} | {term} | {overlap} | {p:.3e} | {q:.3e} | {genes} |".format(
                rank=row.get("rank", ""),
                term=row.get("term", ""),
                overlap=row.get("overlap", ""),
                p=float(row.get("p_value", 1.0)),
                q=float(row.get("q_value", 1.0)),
                genes=", ".join(row.get("overlap_genes", []) or []),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": str(md_path), "csv": str(csv_path)}


def record_target_enrichment_to_action_graph(memory: Any, result: dict[str, Any]) -> None:
    if memory is None or not hasattr(memory, "upsert_node"):
        return
    phenotype = str(result.get("phenotype") or "targets")
    set_id = f"target_enrichment:{slug(phenotype)}"
    memory.upsert_node(
        "target_enrichment_set",
        set_id,
        payload={
            "phenotype": result.get("phenotype"),
            "status": result.get("status"),
            "method": result.get("method"),
            "n_input_genes": result.get("n_input_genes", 0),
        },
        score=0.8 if result.get("status") == "PASS" else 0.3,
    )
    for term in result.get("terms", [])[:100]:
        term_id = f"{set_id}:{slug(term.get('term', 'term'))}"
        score = max(0.0, 1.0 - min(float(term.get("q_value", 1.0)), 1.0))
        memory.upsert_node("enrichment_term", term_id, payload=term, score=score)
        memory.link_nodes("target_enrichment_set", set_id, "enrichment_term", term_id, "contains", weight=score)
        for gene in term.get("overlap_genes", [])[:50]:
            gene = normalize_gene(gene)
            if not gene:
                continue
            memory.upsert_node("gene", gene, payload={"gene": gene}, score=0.6)
            memory.link_nodes(
                "enrichment_term",
                term_id,
                "gene",
                gene,
                relation="includes_gene",
                weight=score,
                evidence={"q_value": term.get("q_value")},
            )


def _source_result(
    source: str,
    status: str,
    *,
    data: Any = None,
    error: str = "",
    url: str = "",
    query: dict[str, Any] | None = None,
    cache_hit: bool = False,
) -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "data": data,
        "error": error,
        "url": url,
        "query": query or {},
        "cache_hit": bool(cache_hit),
        "retrieved_at": _utc_now(),
    }


def _source_metadata(sources: list[str]) -> list[dict[str, str]]:
    return [ANNOTATION_SOURCE_INFO[s] for s in sources if s in ANNOTATION_SOURCE_INFO]


def _empty_target_context(row: dict[str, Any], *, phenotype: str) -> dict[str, Any]:
    return {
        "gene": normalize_gene(row.get("gene")),
        "input": row,
        "phenotype": phenotype,
        "ensembl_id": row.get("ensembl_id", ""),
        "uniprot_id": row.get("uniprot_id", ""),
        "approved_symbol": normalize_gene(row.get("gene")),
        "approved_name": "",
        "protein_name": "",
        "protein_function": "",
        "disease_associations": [],
        "tractability": [],
        "tissue_expression": [],
        "clinical_trials": [],
        "cell_type_context": [],
        "source_status": {},
        "source_results": [],
        "warnings": [],
    }


def _compact_source_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": payload.get("source", ""),
        "status": payload.get("status", ""),
        "url": payload.get("url", ""),
        "cache_hit": bool(payload.get("cache_hit", False)),
        "error": payload.get("error", ""),
        "retrieved_at": payload.get("retrieved_at", ""),
    }


def _annotation_overall_status(targets: list[dict[str, Any]]) -> str:
    statuses = [
        str(status).upper()
        for target in targets
        for status in (target.get("source_status") or {}).values()
    ]
    if not statuses:
        return "NEEDS_INPUT"
    if any(status == "PASS" for status in statuses) and not any(status in {"ERROR", "PARTIAL", "SKIPPED"} for status in statuses):
        return "PASS"
    if any(status == "PASS" for status in statuses):
        return "PARTIAL"
    return "PARTIAL"


def _call_annotation_source(
    source: str,
    client: TargetAnnotationClient,
    context: dict[str, Any],
    *,
    disease_id: str,
    phenotype: str,
    top_n: int,
    cellxgene_snapshot_path: str | Path | None,
) -> dict[str, Any]:
    try:
        if source == "opentargets":
            return _annotate_opentargets(client, context, disease_id=disease_id, top_n=top_n)
        if source == "uniprot":
            return _annotate_uniprot(client, context)
        if source == "gtex":
            return _annotate_gtex(client, context, top_n=top_n)
        if source == "clinicaltrials":
            return _annotate_clinicaltrials(client, context, phenotype=phenotype, top_n=top_n)
        if source == "cellxgene":
            return _annotate_cellxgene(context, snapshot_path=cellxgene_snapshot_path, top_n=top_n)
        return _source_result(source, "SKIPPED", error="Unknown source.")
    except Exception as exc:
        return _source_result(source, "ERROR", error=f"Source parser failed: {exc}")


def _annotate_opentargets(
    client: TargetAnnotationClient,
    context: dict[str, Any],
    *,
    disease_id: str,
    top_n: int,
) -> dict[str, Any]:
    gene = context["gene"]
    ensembl_id = str(context.get("ensembl_id") or "")
    endpoint = "https://api.platform.opentargets.org/api/v4/graphql"
    if not ensembl_id:
        search_query = """
        query Search($queryString: String!) {
          search(queryString: $queryString) {
            hits { id entity name category }
          }
        }
        """
        search_payload = client.request_json(
            "opentargets",
            endpoint,
            method="POST",
            json_body={"query": search_query, "variables": {"queryString": gene}},
        )
        if str(search_payload.get("status")).upper() != "PASS":
            return search_payload
        ensembl_id = _pick_opentargets_gene_id(search_payload.get("data"), gene)
        if not ensembl_id:
            return _source_result("opentargets", "SKIPPED", error=f"No Open Targets gene match for {gene}.", url=endpoint)
        context["ensembl_id"] = ensembl_id

    target_query = """
    query TargetAnnotation($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        approvedName
        tractability { modality value }
        associatedDiseases(page: {index: 0, size: 10}) {
          rows {
            score
            disease { id name }
            datatypeScores { id score }
          }
        }
      }
    }
    """
    payload = client.request_json(
        "opentargets",
        endpoint,
        method="POST",
        json_body={"query": target_query, "variables": {"ensemblId": ensembl_id}},
    )
    if str(payload.get("status")).upper() != "PASS":
        return payload
    if payload.get("data", {}).get("errors"):
        return _source_result("opentargets", "ERROR", error=str(payload["data"]["errors"]), url=endpoint)
    target = ((payload.get("data") or {}).get("data") or {}).get("target") or {}
    if not target:
        return _source_result("opentargets", "SKIPPED", error=f"No target record for {ensembl_id}.", url=endpoint)

    context["ensembl_id"] = target.get("id") or ensembl_id
    context["approved_symbol"] = target.get("approvedSymbol") or gene
    context["approved_name"] = target.get("approvedName") or context.get("approved_name", "")
    context["tractability"] = [
        {"modality": row.get("modality"), "value": bool(row.get("value"))}
        for row in (target.get("tractability") or [])
        if row.get("value")
    ][:10]
    associations = []
    for row in (((target.get("associatedDiseases") or {}).get("rows")) or []):
        disease = row.get("disease") or {}
        if disease_id and disease_id.lower() not in {str(disease.get("id", "")).lower(), str(disease.get("name", "")).lower()}:
            continue
        associations.append({
            "disease_id": disease.get("id", ""),
            "disease_name": disease.get("name", ""),
            "score": float(row.get("score") or 0.0),
            "datatype_scores": row.get("datatypeScores") or [],
        })
    context["disease_associations"] = associations[: max(1, int(top_n or 5))]
    return payload


def _pick_opentargets_gene_id(data: Any, gene: str) -> str:
    hits = (((data or {}).get("data") or {}).get("search") or {}).get("hits") or []
    for hit in hits:
        if hit.get("entity") == "target" and normalize_gene(hit.get("name")) == gene:
            return str(hit.get("id") or "").split(".")[0]
    for hit in hits:
        if hit.get("entity") == "target":
            return str(hit.get("id") or "").split(".")[0]
    return ""


def _annotate_uniprot(client: TargetAnnotationClient, context: dict[str, Any]) -> dict[str, Any]:
    gene = context["gene"]
    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)"
    params = {
        "query": query,
        "fields": "accession,id,gene_names,protein_name,organism_name,reviewed,xref_ensembl,cc_function",
        "format": "json",
        "size": 1,
    }
    payload = client.request_json("uniprot", url, params=params)
    if str(payload.get("status")).upper() != "PASS":
        return payload
    records = (payload.get("data") or {}).get("results") or []
    if not records:
        return _source_result("uniprot", "SKIPPED", error=f"No reviewed human UniProt record for {gene}.", url=url)
    record = records[0]
    context["uniprot_id"] = record.get("primaryAccession") or context.get("uniprot_id", "")
    context["protein_name"] = _uniprot_protein_name(record)
    function_text = _uniprot_function(record)
    if function_text:
        context["protein_function"] = function_text
    if not context.get("ensembl_id"):
        context["ensembl_id"] = _uniprot_ensembl_id(record)
    return payload


def _uniprot_protein_name(record: dict[str, Any]) -> str:
    desc = record.get("proteinDescription") or {}
    recommended = desc.get("recommendedName") or {}
    full = recommended.get("fullName") or {}
    return str(full.get("value") or record.get("uniProtkbId") or "")


def _uniprot_function(record: dict[str, Any]) -> str:
    for comment in record.get("comments") or []:
        if str(comment.get("commentType", "")).upper() != "FUNCTION":
            continue
        texts = [str(t.get("value", "")) for t in comment.get("texts") or [] if t.get("value")]
        if texts:
            return " ".join(texts)[:1000]
    return ""


def _uniprot_ensembl_id(record: dict[str, Any]) -> str:
    for ref in record.get("uniProtKBCrossReferences") or []:
        if ref.get("database") != "Ensembl":
            continue
        ref_id = str(ref.get("id") or "")
        if ref_id.startswith("ENSG"):
            return ref_id.split(".")[0]
        for prop in ref.get("properties") or []:
            value = str(prop.get("value") or "")
            if value.startswith("ENSG"):
                return value.split(".")[0]
    return ""


def _annotate_gtex(client: TargetAnnotationClient, context: dict[str, Any], *, top_n: int) -> dict[str, Any]:
    ensembl_id = str(context.get("ensembl_id") or "").split(".")[0]
    if not ensembl_id:
        return _source_result("gtex", "SKIPPED", error=f"GTEx query requires an Ensembl gene ID for {context['gene']}.")
    url = "https://gtexportal.org/api/v2/expression/medianGeneExpression"
    params = {"gencodeId": ensembl_id, "datasetId": "gtex_v8"}
    payload = client.request_json("gtex", url, params=params)
    if str(payload.get("status")).upper() != "PASS":
        return payload
    rows = _extract_gtex_rows(payload.get("data"))
    rows.sort(key=lambda row: row.get("median_tpm", 0.0), reverse=True)
    context["tissue_expression"] = rows[: max(1, int(top_n or 5))]
    if not rows:
        return _source_result("gtex", "SKIPPED", error=f"No GTEx median expression rows for {ensembl_id}.", url=url)
    return payload


def _extract_gtex_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        rows = data.get("data") or data.get("medianGeneExpression") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tissue = row.get("tissueSiteDetailId") or row.get("tissueSiteDetail") or row.get("tissue") or row.get("smts")
        median = row.get("median") if row.get("median") is not None else row.get("median_tpm", row.get("expression"))
        try:
            median_value = float(median)
        except (TypeError, ValueError):
            median_value = 0.0
        if tissue:
            out.append({"tissue": str(tissue), "median_tpm": median_value, "unit": "TPM"})
    return out


def _annotate_clinicaltrials(
    client: TargetAnnotationClient,
    context: dict[str, Any],
    *,
    phenotype: str,
    top_n: int,
) -> dict[str, Any]:
    gene = context["gene"]
    url = "https://clinicaltrials.gov/api/v2/studies"
    term = " ".join(part for part in [gene, phenotype] if part).strip()
    params = {"query.term": term or gene, "pageSize": max(1, min(int(top_n or 5), 20)), "format": "json"}
    payload = client.request_json("clinicaltrials", url, params=params)
    if str(payload.get("status")).upper() != "PASS":
        return payload
    studies = (payload.get("data") or {}).get("studies") or []
    context["clinical_trials"] = [_parse_clinical_trial(study) for study in studies[: max(1, int(top_n or 5))]]
    return payload


def _parse_clinical_trial(study: dict[str, Any]) -> dict[str, Any]:
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    cond = protocol.get("conditionsModule") or {}
    design = protocol.get("designModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    interventions = []
    for intervention in arms.get("interventions") or []:
        name = intervention.get("name")
        if name:
            interventions.append(str(name))
    phases = design.get("phases") or []
    return {
        "nct_id": ident.get("nctId", ""),
        "brief_title": ident.get("briefTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "conditions": cond.get("conditions") or [],
        "phases": phases,
        "interventions": interventions[:10],
    }


def _annotate_cellxgene(
    context: dict[str, Any],
    *,
    snapshot_path: str | Path | None,
    top_n: int,
) -> dict[str, Any]:
    gene = context["gene"]
    if snapshot_path:
        rows = _load_cellxgene_snapshot(snapshot_path, gene)
        rows.sort(key=lambda row: float(row.get("expression", 0.0)), reverse=True)
        context["cell_type_context"] = rows[: max(1, int(top_n or 5))]
        if rows:
            return _source_result("cellxgene", "PASS", data={"rows": rows}, url=str(snapshot_path))
        return _source_result("cellxgene", "SKIPPED", error=f"No CELLxGENE snapshot rows for {gene}.", url=str(snapshot_path))

    try:
        __import__("cellxgene_census")
    except Exception as exc:
        return _source_result(
            "cellxgene",
            "SKIPPED",
            error=(
                "cellxgene_census is not installed; provide cellxgene_snapshot_path "
                f"or install the optional single-cell dependency. ({exc})"
            ),
        )
    return _source_result(
        "cellxgene",
        "SKIPPED",
        error="Live CELLxGENE Census queries are intentionally disabled unless a bounded local snapshot is supplied.",
    )


def _load_cellxgene_snapshot(path: str | Path, gene: str) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.exists():
        return []
    if path.suffix.lower() in {".json", ".jsonl"}:
        rows = _read_json_records(path)
    else:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = [dict(row) for row in csv.DictReader(fh, delimiter=delimiter)]
    out = []
    for row in rows:
        row_gene = normalize_gene(row.get("gene") or row.get("gene_symbol") or row.get("feature_name"))
        if row_gene != gene:
            continue
        expression = row.get("expression") or row.get("mean_expression") or row.get("fraction_expressing") or 0
        try:
            expression_value = float(expression)
        except (TypeError, ValueError):
            expression_value = 0.0
        out.append({
            "cell_type": row.get("cell_type") or row.get("cellType") or "",
            "tissue": row.get("tissue") or row.get("tissue_general") or "",
            "dataset": row.get("dataset") or row.get("collection_name") or "",
            "expression": expression_value,
        })
    return out


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rows", data.get("data", []))
    return [dict(row) for row in data if isinstance(row, dict)]


def _normalize_gene_inputs(*, gene_list: Any, ranked_genes: Any) -> tuple[list[str], list[tuple[str, float]]]:
    ranked: list[tuple[str, float]] = []
    if ranked_genes:
        for i, row in enumerate(_as_list(ranked_genes), 1):
            if isinstance(row, dict):
                gene = normalize_gene(row.get("gene") or row.get("symbol"))
                score = row.get("score", row.get("rank_score", row.get("stat", 1.0 / i)))
            else:
                gene = normalize_gene(row)
                score = 1.0 / i
            if not gene:
                continue
            try:
                ranked.append((gene, float(score)))
            except (TypeError, ValueError):
                ranked.append((gene, 1.0 / i))
    genes = [normalize_gene(g) for g in _as_list(gene_list) if normalize_gene(g)]
    if not genes and ranked:
        genes = [gene for gene, _ in ranked]
    seen = set()
    unique = []
    for gene in genes:
        if gene not in seen:
            seen.add(gene)
            unique.append(gene)
    if not ranked:
        ranked = [(gene, float(len(unique) - i)) for i, gene in enumerate(unique)]
    return unique, ranked


def _try_gseapy_enrichment(
    *,
    ranked_genes: list[tuple[str, float]],
    genes: list[str],
    gene_sets_path: str | Path | None,
    method: str,
    top_n: int,
    warnings: list[str],
) -> dict[str, Any] | None:
    try:
        import gseapy as gp
    except Exception:
        return None
    try:
        if method == "prerank" and gene_sets_path:
            pre_res = gp.prerank(rnk=ranked_genes, gene_sets=str(gene_sets_path), outdir=None, no_plot=True, seed=7)
            table = getattr(pre_res, "res2d", None)
        elif gene_sets_path:
            enr = gp.enrich(gene_list=genes, gene_sets=str(gene_sets_path), outdir=None, no_plot=True)
            table = getattr(enr, "res2d", None)
        else:
            return None
        if table is None:
            return None
        records = table.to_dict(orient="records") if hasattr(table, "to_dict") else []
        terms = []
        for i, row in enumerate(records[: max(1, int(top_n or 20))], 1):
            term = row.get("Term") or row.get("term") or row.get("Name") or ""
            p = _safe_float(row.get("P-value", row.get("p_value", row.get("NOM p-val", 1.0))), default=1.0)
            q = _safe_float(row.get("Adjusted P-value", row.get("FDR q-val", row.get("q_value", p))), default=p)
            genes_text = row.get("Genes", row.get("Lead_genes", ""))
            overlap_genes = [normalize_gene(g) for g in re.split(r"[;,/ ]+", str(genes_text)) if normalize_gene(g)]
            terms.append({
                "rank": i,
                "term": term,
                "p_value": p,
                "q_value": q,
                "overlap": len(overlap_genes),
                "overlap_genes": overlap_genes,
            })
        return {
            "status": "PASS" if terms else "NO_ENRICHMENT",
            "method": f"gseapy_{method}",
            "gene_sets_path": str(gene_sets_path) if gene_sets_path else "",
            "n_gene_sets": len(terms),
            "terms": terms,
        }
    except Exception as exc:
        warnings.append(f"gseapy failed: {exc}")
        return None


def _run_local_ora(
    genes: list[str],
    gene_sets: dict[str, set[str]],
    universe: list[str],
    *,
    fdr_alpha: float,
    top_n: int,
) -> list[dict[str, Any]]:
    from scipy.stats import hypergeom

    query = set(genes)
    universe_set = set(universe)
    query &= universe_set
    if not query:
        return []
    m = len(universe_set)
    n = len(query)
    rows = []
    for term, term_genes in gene_sets.items():
        term_set = set(term_genes) & universe_set
        overlap = sorted(query & term_set)
        k = len(overlap)
        if k == 0:
            continue
        p_value = float(hypergeom.sf(k - 1, m, len(term_set), n))
        rows.append({
            "term": term,
            "p_value": p_value,
            "overlap": k,
            "set_size": len(term_set),
            "query_size": n,
            "overlap_genes": overlap,
        })
    q_values = benjamini_hochberg([row["p_value"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value"] = float(q_value)
        row["significant"] = bool(q_value <= fdr_alpha)
    rows.sort(key=lambda row: (row["q_value"], row["p_value"], -row["overlap"], row["term"]))
    for i, row in enumerate(rows[: max(1, int(top_n or 20))], 1):
        row["rank"] = i
    return rows[: max(1, int(top_n or 20))]


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running = 1.0
    for rank, idx in enumerate(reversed(order), 1):
        original_rank = n - rank + 1
        p = max(min(float(p_values[idx]), 1.0), 0.0)
        running = min(running, p * n / original_rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _enrichment_caveats() -> list[str]:
    return [
        "Gene-set enrichment is exploratory and depends on the supplied gene universe and GMT provenance.",
        "Pathway enrichment does not establish mechanism without orthogonal functional evidence.",
        "Enrichment context must not override direct biobank genetic or epidemiological evidence.",
    ]
