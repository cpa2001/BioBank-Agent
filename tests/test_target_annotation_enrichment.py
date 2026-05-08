from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from biobank_agent.data.target_context import (
    build_target_annotation_context,
    build_target_enrichment,
    load_gene_sets,
)
from biobank_agent.registry import autodiscover_skills, get_registry
from biobank_agent.skills.report import _extract_key_findings, _interpret_skill
from biobank_agent.skills.safety_check import safety_check
from biobank_agent.skills.statistical_review import statistical_review
from biobank_agent.skills.target_annotation_context import target_annotation_context
from biobank_agent.skills.target_enrichment import target_enrichment
from biobank_agent.state import AnalysisRecord, SessionState


class RecordingMemory:
    def __init__(self) -> None:
        self.upserts = []
        self.links = []

    def upsert_node(self, *args, **kwargs):
        self.upserts.append((args, kwargs))
        return ":".join(str(a) for a in args[:2])

    def link_nodes(self, *args, **kwargs):
        self.links.append((args, kwargs))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, params=None, json=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json})
        if "opentargets" in url and json and "Search" in json.get("query", ""):
            return FakeResponse({
                "data": {
                    "search": {
                        "hits": [{"id": "ENSG00000130164", "entity": "target", "name": "LDLR", "category": ["protein_coding"]}]
                    }
                }
            })
        if "opentargets" in url:
            return FakeResponse({
                "data": {
                    "target": {
                        "id": "ENSG00000130164",
                        "approvedSymbol": "LDLR",
                        "approvedName": "low density lipoprotein receptor",
                        "tractability": [{"modality": "SM", "value": True}],
                        "associatedDiseases": {
                            "rows": [
                                {
                                    "score": 0.91,
                                    "disease": {"id": "EFO_0004611", "name": "hypercholesterolemia"},
                                    "datatypeScores": [{"id": "genetic_association", "score": 0.9}],
                                }
                            ]
                        },
                    }
                }
            })
        if "uniprot" in url:
            return FakeResponse({
                "results": [
                    {
                        "primaryAccession": "P01130",
                        "uniProtkbId": "LDLR_HUMAN",
                        "proteinDescription": {"recommendedName": {"fullName": {"value": "Low-density lipoprotein receptor"}}},
                        "comments": [{"commentType": "FUNCTION", "texts": [{"value": "Receptor for plasma LDL."}]}],
                    }
                ]
            })
        if "gtexportal" in url:
            return FakeResponse({
                "data": [
                    {"tissueSiteDetailId": "Liver", "median": 42.0},
                    {"tissueSiteDetailId": "Adipose_Subcutaneous", "median": 4.0},
                ]
            })
        if "clinicaltrials" in url:
            return FakeResponse({
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "LDLR lipid study"},
                            "statusModule": {"overallStatus": "COMPLETED"},
                            "conditionsModule": {"conditions": ["Hypercholesterolemia"]},
                            "designModule": {"phases": ["PHASE2"]},
                            "armsInterventionsModule": {"interventions": [{"name": "lipid lowering"}]},
                        }
                    }
                ]
            })
        raise AssertionError(f"Unexpected URL: {url}")


def _gmt(path: Path) -> Path:
    path.write_text(
        "lipid_transport\tna\tLDLR\tAPOB\tPCSK9\tLDLRAP1\n"
        "immune_noise\tna\tIL6\tTNF\tCXCL8\tCCR5\n",
        encoding="utf-8",
    )
    return path


def test_target_annotation_context_queries_sources_and_reuses_cache(tmp_path: Path, monkeypatch):
    from biobank_agent.data import target_context as mod

    FakeClient.calls = []
    monkeypatch.setattr(mod.httpx, "Client", FakeClient)
    cache_dir = tmp_path / "cache"
    first = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["opentargets", "uniprot", "gtex", "clinicaltrials"],
        cache_dir=cache_dir,
        top_n=3,
    )

    assert first["status"] == "PASS"
    target = first["targets"][0]
    assert target["ensembl_id"] == "ENSG00000130164"
    assert target["uniprot_id"] == "P01130"
    assert target["disease_associations"][0]["disease_name"] == "hypercholesterolemia"
    assert target["tissue_expression"][0]["tissue"] == "Liver"
    assert target["clinical_trials"][0]["nct_id"] == "NCT00000001"
    assert len(FakeClient.calls) == 5

    class FailingClient(FakeClient):
        def request(self, *args, **kwargs):  # pragma: no cover - should not be reached
            raise AssertionError("cache_first should avoid live HTTP calls")

    monkeypatch.setattr(mod.httpx, "Client", FailingClient)
    second = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["opentargets", "uniprot", "gtex", "clinicaltrials"],
        cache_dir=cache_dir,
        top_n=3,
    )
    assert second["targets"][0]["source_results"][0]["cache_hit"] is True


def test_target_annotation_context_skill_writes_artifacts_and_records_memory(tmp_path: Path):
    snapshot = tmp_path / "cellxgene.csv"
    snapshot.write_text(
        "gene,cell_type,tissue,expression,dataset\nLDLR,hepatocyte,liver,12.5,local_snapshot\n",
        encoding="utf-8",
    )
    memory = RecordingMemory()
    ctx = SimpleNamespace(report_dir=tmp_path, memory=memory)

    result = target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["cellxgene"],
        cellxgene_snapshot_path=str(snapshot),
        ctx=ctx,
    )

    assert result["status"] == "PASS"
    assert result["targets"][0]["cell_type_context"][0]["cell_type"] == "hepatocyte"
    assert Path(result["artifacts"]["markdown"]).exists()
    assert any(call[0][0] == "target_annotation_context" for call in memory.upserts)
    assert any(call[1].get("relation") == "supported_by_external_source" for call in memory.links)


def test_target_annotation_context_skips_identifier_dependent_sources_without_network():
    result = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["gtex", "cellxgene"],
        source_mode="cache_only",
        cache_dir=None,
    )

    assert result["status"] == "PARTIAL"
    warnings = "\n".join(result["warnings"])
    assert "GTEx query requires an Ensembl gene ID" in warnings
    assert "cellxgene_census is not installed" in warnings or "CELLxGENE" in warnings


def test_target_annotation_context_rejects_unknown_sources_and_bad_top_n():
    result = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["unknown_source"],
        top_n="not-an-int",
    )

    assert result["status"] == "INVALID_INPUT"
    assert result["invalid_sources"] == ["unknown_source"]
    assert "opentargets" in result["supported_sources"]


def test_target_annotation_context_source_parser_errors_are_source_local(tmp_path: Path, monkeypatch):
    from biobank_agent.data import target_context as mod

    class MalformedOpenTargets(FakeClient):
        def request(self, method, url, params=None, json=None):
            if "opentargets" in url and json and "Search" in json.get("query", ""):
                return FakeResponse({
                    "data": {
                        "search": {
                            "hits": [{"id": "ENSG00000130164", "entity": "target", "name": "LDLR"}]
                        }
                    }
                })
            if "opentargets" in url:
                return FakeResponse({
                    "data": {
                        "target": {
                            "id": "ENSG00000130164",
                            "approvedSymbol": "LDLR",
                            "associatedDiseases": {
                                "rows": [{"score": "not-a-number", "disease": {"id": "EFO_1", "name": "bad score"}}]
                            },
                        }
                    }
                })
            return super().request(method, url, params=params, json=json)

    monkeypatch.setattr(mod.httpx, "Client", MalformedOpenTargets)
    result = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["opentargets", "uniprot"],
        cache_dir=tmp_path / "cache",
    )

    assert result["status"] == "PARTIAL"
    assert result["targets"][0]["source_status"]["opentargets"] == "ERROR"
    assert result["targets"][0]["source_status"]["uniprot"] == "PASS"
    assert "Source parser failed" in "\n".join(result["warnings"])


def test_target_enrichment_local_gmt_and_memory(tmp_path: Path):
    gmt = _gmt(tmp_path / "sets.gmt")
    memory = RecordingMemory()
    ctx = SimpleNamespace(report_dir=tmp_path, memory=memory)

    result = target_enrichment(
        gene_list=["LDLR", "APOB", "PCSK9", "LDLR"],
        phenotype="LDL cholesterol",
        gene_sets_path=str(gmt),
        universe_genes=["LDLR", "APOB", "PCSK9", "LDLRAP1", "IL6", "TNF", "CXCL8", "CCR5"],
        ctx=ctx,
    )

    assert result["status"] == "PASS"
    assert result["terms"][0]["term"] == "lipid_transport"
    assert result["terms"][0]["overlap_genes"] == ["APOB", "LDLR", "PCSK9"]
    assert result["explicit_universe"] is True
    assert Path(result["artifacts"]["csv"]).exists()
    assert any(call[0][0] == "target_enrichment_set" for call in memory.upserts)
    assert any(call[1].get("relation") == "includes_gene" for call in memory.links)


def test_target_enrichment_needs_local_gene_sets_and_parses_gmt(tmp_path: Path):
    assert load_gene_sets(_gmt(tmp_path / "sets.gmt"))["lipid_transport"] == {"LDLR", "APOB", "PCSK9", "LDLRAP1"}

    result = build_target_enrichment(gene_list=["LDLR", "APOB"], phenotype="LDL cholesterol")

    assert result["status"] == "NEEDS_INPUT"
    assert "GMT" in result["message"]


def test_target_enrichment_handles_missing_gmt_and_dropped_universe_genes(tmp_path: Path):
    missing = build_target_enrichment(
        gene_list=["LDLR", "APOB"],
        phenotype="LDL cholesterol",
        gene_sets_path=str(tmp_path / "missing.gmt"),
        top_n="bad",
        fdr_alpha="bad",
    )
    assert missing["status"] == "INVALID_INPUT"
    assert "Could not read" in missing["message"]

    result = build_target_enrichment(
        gene_list=["LDLR", "APOB", "PCSK9"],
        phenotype="LDL cholesterol",
        gene_sets_path=str(_gmt(tmp_path / "sets.gmt")),
        universe_genes=["LDLR", "APOB", "IL6", "TNF", "CXCL8", "CCR5"],
        top_n="bad",
        fdr_alpha=2,
    )
    assert result["fdr_alpha"] == 0.05
    assert result["n_tested_genes"] == 2
    assert result["dropped_genes"] == ["PCSK9"]
    assert any("outside the enrichment universe" in warning for warning in result["warnings"])


def test_target_enrichment_prerank_falls_back_when_gseapy_missing(tmp_path: Path, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gseapy":
            raise ModuleNotFoundError("No module named gseapy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = target_enrichment(
        ranked_genes=[{"gene": "LDLR", "score": 4}, {"gene": "APOB", "score": 3}, {"gene": "PCSK9", "score": 2}],
        phenotype="LDL cholesterol",
        gene_sets_path=str(_gmt(tmp_path / "sets.gmt")),
        method="prerank",
        write_report=False,
    )

    assert result["method"] == "local_ora"
    assert any("gseapy" in warning for warning in result["warnings"])


def test_report_statistical_review_and_safety_understand_target_context(tmp_path: Path):
    annotation = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["gtex"],
        source_mode="cache_only",
    )
    enrichment = build_target_enrichment(
        gene_list=["LDLR", "APOB", "PCSK9"],
        phenotype="LDL cholesterol",
        gene_sets_path=str(_gmt(tmp_path / "sets.gmt")),
    )
    records = [
        AnalysisRecord("2026-05-08T00:00:00", "target_annotation_context", {}, annotation, []),
        AnalysisRecord("2026-05-08T00:01:00", "target_enrichment", {}, enrichment, []),
    ]
    ctx = SimpleNamespace(state=SessionState(records=records))

    assert "Target annotation context" in _interpret_skill(records[0])
    assert "Target enrichment" in _interpret_skill(records[1])
    findings = _extract_key_findings(records)
    assert any("Target enrichment" in finding for finding in findings)

    stats = statistical_review(scope="session", ctx=ctx)
    issue_types = {issue["type"] for issue in stats["issues"]}
    assert "target_annotation_partial_sources" in issue_types
    assert "target_annotation_missing_gene_id" in issue_types
    assert "target_enrichment_missing_universe" in issue_types

    safety = safety_check(scope="session", ctx=ctx)
    safety_types = {issue["type"] for issue in safety["issues"]}
    assert "external_annotation_license_review" in safety_types
    assert "enrichment_context_release_review" in safety_types


def test_statistical_review_flags_invalid_annotation_sources():
    invalid = build_target_annotation_context(
        targets=[{"gene": "LDLR"}],
        phenotype="LDL cholesterol",
        sources=["bad_source"],
    )
    record = AnalysisRecord("2026-05-08T00:02:00", "target_annotation_context", {}, invalid, [])
    stats = statistical_review(scope="session", ctx=SimpleNamespace(state=SessionState(records=[record])))

    assert invalid["status"] == "INVALID_INPUT"
    assert any(issue["type"] == "target_annotation_invalid_input" for issue in stats["issues"])


def test_target_annotation_and_enrichment_skills_are_registered():
    autodiscover_skills()
    names = {s["name"] for s in get_registry().list_skills()}
    assert "target_annotation_context" in names
    assert "target_enrichment" in names

    schemas = {
        s["function"]["name"]: s["function"]["parameters"]["properties"]
        for s in get_registry().tool_schemas()
        if s["function"]["name"] in {"target_annotation_context", "target_enrichment"}
    }
    assert "cellxgene_snapshot_path" in schemas["target_annotation_context"]
    assert "gene_sets_path" in schemas["target_enrichment"]
