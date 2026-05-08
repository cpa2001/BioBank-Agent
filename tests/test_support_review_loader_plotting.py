"""Tests for statistical review, data loader, web search, platform, and plotting utilities."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biobank_agent.data import loader as loader_mod
from biobank_agent.skills import statistical_review as review_mod
from biobank_agent.skills import web_search as web_search_mod
from biobank_agent.utils import platform as platform_mod
from biobank_agent.utils import plotting as plotting_mod


def record(skill, args=None, key_results=None):
    return SimpleNamespace(skill=skill, args=args or {}, key_results=key_results or {})


def test_statistical_review_empty_scopes_and_issue_sorting():
    ctx = SimpleNamespace(state=SimpleNamespace(records=[]))
    assert review_mod.statistical_review(ctx=ctx) == {"issues": [], "message": "No analyses to review."}

    records = [
        record("train_model", {"model_key": "E11_xgb"}, {"mean_auc": 0.99, "n_cases": 10, "n_total": 1000}),
        record("train_model", {"model_key": "I10_xgb"}, {"mean_auc": 0.52, "n_cases": 500, "n_total": 1000}),
        record("phewas", {}, {"n_fields_tested": 150}),
        record("survival", {}, {"n_events": 20}),
        record("correlation", {}, {"max_correlation": 0.91}),
        record("cohort_card", {}, {"status": "PARTIAL", "n_cases": 50, "bias_flags": ["immortal time risk"]}),
        record("trajectory_tokenize", {}, {"n_tokens": 0, "n_participants": 5}),
        record("world_model_audit", {}, {"safety_status": "FAIL", "allowed_claim_type": "association"}),
        record("world_model_audit", {}, {"safety_status": "PARTIAL", "allowed_claim_type": "forecast"}),
    ]
    ctx.state.records = records

    session = review_mod.statistical_review(ctx=ctx)
    last = review_mod.statistical_review(scope="last", ctx=ctx)
    model = review_mod.statistical_review(scope="model:E11_xgb", ctx=ctx)
    clean = review_mod.statistical_review(
        ctx=SimpleNamespace(state=SimpleNamespace(records=[record("train_model", {}, {"mean_auc": 0.8, "n_cases": 500, "n_total": 1000})]))
    )

    assert session["overall_assessment"] == "CRITICAL — review required before publication"
    assert session["n_critical"] >= 5
    assert session["issues"][0]["severity"] == "CRITICAL"
    assert {issue["type"] for issue in session["issues"]} >= {
        "data_leakage",
        "class_imbalance",
        "small_sample",
        "poor_discrimination",
        "multiple_testing",
        "low_event_count",
        "multicollinearity",
        "cohort_card_partial",
        "small_cross_cohort_case_count",
        "design_bias_check_required",
        "empty_trajectory_dataset",
        "unsupported_world_model_claim",
        "partial_world_model_evidence",
    }
    assert last["n_records_reviewed"] == 1
    assert last["issues"][0]["type"] == "partial_world_model_evidence"
    assert model["n_records_reviewed"] == 1
    assert any(issue["type"] == "data_leakage" for issue in model["issues"])
    assert clean["overall_assessment"] == "CLEAN — no statistical issues detected"

    caution = review_mod.statistical_review(
        ctx=SimpleNamespace(state=SimpleNamespace(records=[
            record("train_model", {}, {"mean_auc": 0.54, "n_cases": 500, "n_total": 1000}),
            record("survival", {}, {"n_events": 20}),
            record("cohort_card", {}, {"status": "PARTIAL", "n_cases": 500}),
        ]))
    )
    assert caution["overall_assessment"] == "CAUTION — several issues to address"

    branch_clean = review_mod.statistical_review(
        ctx=SimpleNamespace(state=SimpleNamespace(records=[
            record("train_model", {}, {"mean_auc": "not numeric", "n_cases": "unknown", "n_total": 0}),
            record("phewas", {"correction": "fdr"}, {"n_fields_tested": 50}),
            record("survival", {}, {"n_events": 100}),
            record("correlation", {}, {"max_correlation": 0.2}),
            record("cohort_card", {}, {"status": "READY", "n_cases": 500, "bias_flags": []}),
            record("trajectory_tokenize", {}, {"n_tokens": 20, "n_participants": 5}),
            record("world_model_audit", {}, {"safety_status": "PASS"}),
        ]))
    )
    assert branch_clean["overall_assessment"] == "CLEAN — no statistical issues detected"


def make_loader_settings(tmp_path):
    return SimpleNamespace(
        biomarker_parquet=tmp_path / "missing_biomarkers",
        diagnoses_parquet=tmp_path / "missing_diagnoses",
        deaths_parquet=tmp_path / "missing_deaths",
        category_parquet_dir=tmp_path / "missing_categories",
        raw_csv_dir=tmp_path,
        subject_id_col="eid",
        diagnoses_code_col="diag_icd10",
        deaths_code_col="cause_icd10",
    )


def test_data_manager_field_routing_queries_csv_and_refresh(tmp_path):
    loader_mod.DataManager._FIELD_TO_CSV = None
    settings = make_loader_settings(tmp_path)
    dm = loader_mod.DataManager(settings)
    dm.conn.execute('CREATE TABLE biomarkers AS SELECT * FROM (VALUES (1, 7.1, 22.0), (2, 5.2, 28.0)) AS t(eid, "30740-0.0", "21001-0.0")')
    dm.conn.execute("CREATE TABLE diagnoses AS SELECT * FROM (VALUES (1, 'E11'), (2, 'I10')) AS t(eid, diag_icd10)")
    dm.conn.execute("CREATE TABLE deaths AS SELECT * FROM (VALUES (2, 'I25')) AS t(eid, cause_icd10)")
    dm.conn.execute('CREATE TABLE cat_demo AS SELECT * FROM (VALUES (1, 44), (2, 55)) AS t(eid, "999-0.0")')
    dm._category_views = ["cat_demo"]
    dm._category_field_map = {"999": "cat_demo"}

    (tmp_path / "Population_Characteristics.txt").write_text("123\n", encoding="utf-8")
    (tmp_path / "ukb672073_Population_Characteristics.csv").write_text('eid,"123-0.0"\n1,9\n2,10\n', encoding="utf-8")

    assert loader_mod._escape_path(Path("a'b")) == "a''b"
    assert dm.data_available is True
    assert dm.count_subjects() == 2
    assert dm.list_parquet_columns() == ["eid", "30740-0.0", "21001-0.0"]
    assert dm.field_source("30740") == "parquet"
    assert dm.field_source("999") == "cat_demo"
    assert dm.field_source("123") == "Population_Characteristics"
    assert dm.field_source("nope") == "unknown"

    parquet_field = dm.get_field("30740")
    category_field = dm.get_field("999", eids=[2])
    csv_field = dm.get_field("123")
    matrix = dm.get_biomarker_matrix(["30740", "21001"], eids=[1])
    matrix_all = dm.get_biomarker_matrix(["30740"])
    diagnoses = dm.get_diagnoses("E")
    all_diagnoses = dm.get_diagnoses()
    deaths = dm.get_deaths()
    filtered_deaths = dm.get_deaths("I")

    assert parquet_field["value"].tolist() == [7.1, 5.2]
    assert category_field["value"].tolist() == [55]
    assert csv_field["value"].tolist() == [9, 10]
    assert matrix["30740"].tolist() == [7.1]
    assert matrix_all["30740"].tolist() == [7.1, 5.2]
    assert diagnoses["diag_icd10"].tolist() == ["E11"]
    assert all_diagnoses["diag_icd10"].tolist() == ["E11", "I10"]
    assert deaths["cause_icd10"].tolist() == ["I25"]
    assert filtered_deaths["cause_icd10"].tolist() == ["I25"]
    with pytest.raises(ValueError, match="Field missing"):
        dm.get_field("missing")

    dm._register_csv_view("Population_Characteristics")
    dm._register_csv_view("Not_A_Category")
    dm.refresh_parquet_views()
    assert dm._parquet_fields is None


def test_data_manager_init_registers_parquet_file_and_handles_unavailable(tmp_path):
    pytest.importorskip("pyarrow")
    biomarker_path = tmp_path / "biomarkers.parquet"
    pd.DataFrame({"eid": [1], "30740-0.0": [5.0]}).to_parquet(biomarker_path)
    settings = make_loader_settings(tmp_path)
    settings.biomarker_parquet = biomarker_path
    dm = loader_mod.DataManager(settings)

    assert dm.data_available is True
    assert dm.get_field("30740")["value"].tolist() == [5.0]

    empty = loader_mod.DataManager(make_loader_settings(tmp_path / "other"))
    assert empty.data_available is False
    assert empty._get_parquet_fields() == set()

    biomarker_dir = tmp_path / "biomarker_parts"
    biomarker_dir.mkdir()
    pd.DataFrame({"eid": [1], "30870-0.0": [2.0]}).to_parquet(biomarker_dir / "part.parquet")
    dir_settings = make_loader_settings(tmp_path)
    dir_settings.biomarker_parquet = biomarker_dir
    dir_dm = loader_mod.DataManager(dir_settings)
    assert dir_dm.get_field("30870")["value"].tolist() == [2.0]


def test_data_manager_category_parquet_and_defensive_loader_branches(tmp_path):
    pytest.importorskip("pyarrow")
    loader_mod.DataManager._FIELD_TO_CSV = None

    category_dir = tmp_path / "categories"
    category_dir.mkdir()
    pd.DataFrame({"eid": [1], "777-0.0": [12.0], "abc-0.0": [99.0]}).to_parquet(category_dir / "demo.parquet")
    (category_dir / "broken.parquet").write_text("not parquet", encoding="utf-8")

    settings = make_loader_settings(tmp_path)
    settings.category_parquet_dir = category_dir
    dm = loader_mod.DataManager(settings)

    assert dm.field_source("777").startswith("cat_")
    assert dm.get_field("777")["value"].tolist() == [12.0]

    dm._parquet_fields = None
    dm._category_views.append("missing_view")
    assert "777" in dm._get_parquet_fields()

    missing_csv = loader_mod.DataManager(make_loader_settings(tmp_path / "missing_csv"))
    missing_csv._register_csv_view("Population_Characteristics")
    missing_csv._register_csv_view("Genomics")

    (tmp_path / "Population_Characteristics.txt").write_text("456\n   \n789\n", encoding="utf-8")
    loader_mod.DataManager._FIELD_TO_CSV = None
    mapping = dm._build_field_csv_map()
    assert mapping["456"] == "Population_Characteristics"
    assert mapping["789"] == "Population_Characteristics"


def test_data_manager_parquet_field_introspection_exceptions(tmp_path):
    dm = loader_mod.DataManager(make_loader_settings(tmp_path))

    class AlwaysRaisingConn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("duckdb unavailable")

    dm.conn = AlwaysRaisingConn()
    dm._parquet_fields = None
    assert dm._get_parquet_fields() == set()

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class RaisingOnCategoryConn:
        def __init__(self):
            self.calls = 0

        def execute(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Result([("eid",), ("30740-0.0",)])
            raise RuntimeError("category view unavailable")

    dm.conn = RaisingOnCategoryConn()
    dm._category_views = ["cat_missing"]
    dm._parquet_fields = None
    assert dm._get_parquet_fields() == {"30740"}


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_web_search_backends_and_provider_selection(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results=10):
            return [{"title": "Duck", "href": "https://d", "body": "snippet"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=lambda: FakeDDGS()))
    assert web_search_mod._search_duckduckgo("q", 2) == [{"title": "Duck", "url": "https://d", "snippet": "snippet"}]

    class RaisingDDGS:
        def text(self, query, max_results=10):
            raise RuntimeError("rate limited")

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=lambda: RaisingDDGS()))
    limited = web_search_mod._search_duckduckgo("q", 2)
    assert "Rate limited" in limited[0]["title"]

    def fake_get(url, headers=None, params=None, timeout=15):
        assert params["count"] == 2
        return FakeResponse({"web": {"results": [{"title": "Brave", "url": "https://b", "description": "desc"}]}})

    def fake_post(url, headers=None, json=None, timeout=15):
        assert json["num"] == 2
        return FakeResponse({"organic": [{"title": "Serper", "link": "https://s", "snippet": "snip"}]})

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get, post=fake_post))
    assert web_search_mod._search_brave("q", 2, "key")[0]["title"] == "Brave"
    assert web_search_mod._search_serper("q", 2, "key")[0]["title"] == "Serper"

    monkeypatch.setattr(web_search_mod, "_search_brave", lambda q, m, k: [{"title": "brave"}])
    monkeypatch.setattr(web_search_mod, "_search_serper", lambda q, m, k: [{"title": "serper"}])
    monkeypatch.setattr(web_search_mod, "_search_duckduckgo", lambda q, m: [{"title": "ddg"}])

    brave_ctx = SimpleNamespace(settings=SimpleNamespace(search_provider="brave", search_api_key="key"))
    serper_ctx = SimpleNamespace(settings=SimpleNamespace(search_provider="serper", search_api_key="key"))
    missing_key_ctx = SimpleNamespace(settings=SimpleNamespace(search_provider="brave", search_api_key=""))
    unknown_ctx = SimpleNamespace(settings=SimpleNamespace(search_provider="unknown", search_api_key="key"))

    assert web_search_mod.web_search("q", max_results=2, ctx=brave_ctx)["results"][0]["title"] == "brave"
    assert web_search_mod.web_search("q", max_results=2, ctx=serper_ctx)["results"][0]["title"] == "serper"
    assert web_search_mod.web_search("q", max_results=2, ctx=missing_key_ctx)["results"][0]["title"] == "ddg"
    assert web_search_mod.web_search("q", max_results=2, ctx=unknown_ctx)["results"][0]["title"] == "ddg"

    monkeypatch.setattr(web_search_mod, "_search_duckduckgo", lambda q, m: (_ for _ in ()).throw(RuntimeError("offline")))
    error = web_search_mod.web_search("q", ctx=unknown_ctx)
    assert error["error"] == "Search failed: offline"


def test_web_search_duckduckgo_legacy_import_and_partial_error(monkeypatch):
    real_import = builtins.__import__

    class FakeDDGS:
        def text(self, query, max_results=10):
            return [{"title": "Legacy", "link": "https://legacy", "snippet": "old"}]

    def legacy_import(name, *args, **kwargs):
        if name == "ddgs":
            raise ImportError("new package missing")
        if name == "duckduckgo_search":
            return SimpleNamespace(DDGS=lambda: FakeDDGS())
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", legacy_import)
    assert web_search_mod._search_duckduckgo("q", 1) == [
        {"title": "Legacy", "url": "https://legacy", "snippet": "old"}
    ]

    class PartialDDGS:
        def text(self, query, max_results=10):
            yield {"title": "One", "href": "https://one", "body": "snippet"}
            raise RuntimeError("rate limit after one")

    monkeypatch.setattr(builtins, "__import__", lambda name, *args, **kwargs: SimpleNamespace(DDGS=lambda: PartialDDGS()) if name == "ddgs" else real_import(name, *args, **kwargs))
    assert web_search_mod._search_duckduckgo("q", 2) == [
        {"title": "One", "url": "https://one", "snippet": "snippet"}
    ]


def test_platform_gpu_memory_and_summary_paths(monkeypatch):
    real_import = builtins.__import__

    def no_gpu_import(name, *args, **kwargs):
        if name in {"torch", "cupy"}:
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_gpu_import)
    assert platform_mod.has_gpu() is False
    assert platform_mod.gpu_info() == {"available": False}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(index):
            return "Fake GPU"

        @staticmethod
        def get_device_properties(index):
            return SimpleNamespace(total_mem=12_000_000_000)

    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=64_000_000_000, available=32_000_000_000)),
    )

    gpu = platform_mod.gpu_info()
    info = platform_mod.platform_info()
    summary = platform_mod.platform_summary()

    assert gpu["device_name"] == "Fake GPU"
    assert gpu["memory_gb"] == 12.0
    assert info["memory_total_gb"] == 64.0
    assert "GPU: Fake GPU" in summary

    class BrokenCuda(FakeCuda):
        @staticmethod
        def device_count():
            raise RuntimeError("driver issue")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=BrokenCuda))
    assert platform_mod.gpu_info() == {"available": True, "device_count": 1, "detail": "unknown"}

    def cupy_only_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        if name == "cupy":
            return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1)))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cupy_only_import)
    assert platform_mod.has_gpu() is True

    def no_psutil_or_gpu_import(name, *args, **kwargs):
        if name in {"torch", "cupy", "psutil"}:
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil_or_gpu_import)
    no_mem = platform_mod.platform_info()
    assert "memory_total_gb" not in no_mem
    assert "No GPU" in platform_mod.platform_summary()


def test_plotting_helpers_save_legend_heatmap_and_brackets(tmp_path):
    fig, axes = plotting_mod.nature_figure(nrows=1, ncols=2, width="double")
    plotting_mod.add_panel_labels(fig, axes)
    axes[0].plot([0, 1], [0, 1], label="a")
    axes[0].plot([0, 1], [1, 0], label="b")
    inside = plotting_mod.smart_legend(axes[0])
    assert inside is not None

    for i in range(6):
        axes[1].plot([0, 1], [i, i], label=f"line-{i}")
    outside = plotting_mod.smart_legend(axes[1])
    assert outside is not None

    paths = plotting_mod.save_figure(fig, "test_plot", tmp_path, formats=("png",))
    assert paths == [tmp_path / "test_plot.png"]
    assert paths[0].exists()

    fig2, ax2 = plotting_mod.icml_figure()
    im = plotting_mod.publication_heatmap(
        np.array([[1.0, np.nan], [-2.0, 3.0]]),
        ax2,
        center=0,
        cbar=True,
        cbar_label="effect",
        xticklabels=["x1", "x2"],
        yticklabels=["y1", "y2"],
        annot=True,
    )
    assert im.get_array().shape == (2, 2)
    plotting_mod.add_significance_bracket(ax2, 0, 1, 3.2, p_value=0.02)
    plotting_mod.add_significance_bracket(ax2, 0, 1, 3.5, p_value=0.5)
    no_legend = plotting_mod.smart_legend(ax2)
    assert no_legend is not None
    plt.close("all")


def test_plotting_invisible_panel_heatmap_defaults_and_strong_brackets():
    fig, axes = plotting_mod.nature_figure(nrows=1, ncols=2)
    axes[1].set_visible(False)
    plotting_mod.add_panel_labels(fig, axes, labels=["A", "B"])
    assert [text.get_text() for text in axes[0].texts] == ["A"]
    assert len(axes[1].texts) == 0

    fig2, ax2 = plotting_mod.nature_figure()
    im = plotting_mod.publication_heatmap(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        ax2,
        cbar=False,
        annot=True,
        fmt=".1f",
    )
    assert im.get_clim() == (1.0, 4.0)

    plotting_mod.add_significance_bracket(ax2, 0, 1, 4.5, p_value=0.0005)
    plotting_mod.add_significance_bracket(ax2, 0, 1, 4.8, p_value=0.005)
    labels = [text.get_text() for text in ax2.texts]
    assert "***" in labels
    assert "**" in labels

    fig3, ax3 = plotting_mod.nature_figure()
    plotting_mod.add_panel_labels(fig3, ax3)
    assert ax3.texts[0].get_text() == "(a)"

    fig4, axes4 = plotting_mod.nature_figure(nrows=1, ncols=2)
    plotting_mod.add_panel_labels(fig4, list(axes4), labels=["left", "right"])
    assert [ax.texts[0].get_text() for ax in axes4] == ["left", "right"]
    plt.close("all")


def test_publication_heatmap_optional_branches_without_labels_or_annotations():
    fig, ax = plotting_mod.nature_figure()
    im = plotting_mod.publication_heatmap(
        np.array([[0.0, 1.0], [np.nan, 4.0]]),
        ax,
        vmin=-1.0,
        vmax=5.0,
        center=None,
        cbar=True,
        cbar_label=None,
        xticklabels=None,
        yticklabels=None,
        annot=False,
    )

    assert im.get_clim() == (-1.0, 5.0)
    assert len(ax.texts) == 0
    plt.close("all")
