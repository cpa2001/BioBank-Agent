"""Tests for PheWAS, survival, embedding, and GWAS-proxy skills."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biobank_agent.skills import embedding as embedding_mod
from biobank_agent.skills import gwas_proxy as gwas_mod
from biobank_agent.skills import phewas as phewas_mod
from biobank_agent.skills import survival as survival_mod


class AnalysisState:
    def __init__(self):
        self.figures = []
        self.feature_matrix = None
        self.labels = None


def analysis_ctx(tmp_path, dm=None):
    return SimpleNamespace(
        report_dir=tmp_path,
        dm=dm,
        state=AnalysisState(),
        catalog=SimpleNamespace(fields={"30740": {"title": "Glucose"}, "30750": {"title": "HbA1c"}}),
        settings=SimpleNamespace(
            subject_id_col="eid",
            diagnoses_code_col="diag_icd10",
            deaths_code_col="cause_icd10",
        ),
    )


def fake_save_figure(fig, name, report_dir, formats=("png", "svg")):
    return [report_dir / f"{name}.{fmt}" for fmt in formats]


class FakePhewasDM:
    def __init__(self):
        self.values = pd.DataFrame(
            {
                "eid": np.arange(1, 301),
                "value": np.r_[np.linspace(8, 10, 120), np.linspace(4, 6, 180)],
            }
        )

    def query(self, sql, params=None):
        if "FROM biomarkers" in sql:
            return self.values
        if "GROUP BY code" in sql:
            return pd.DataFrame({"code": ["E11", "I10"], "n": [120, 110]})
        if "LIKE 'E11%'" in sql:
            return pd.DataFrame({"eid": np.arange(1, 121)})
        if "LIKE 'I10%'" in sql:
            return pd.DataFrame({"eid": np.arange(31, 141)})
        raise AssertionError(sql)


def test_phewas_success_and_no_results_path(tmp_path, monkeypatch):
    ctx = analysis_ctx(tmp_path, dm=FakePhewasDM())
    monkeypatch.setattr(phewas_mod, "save_figure", fake_save_figure)

    result = phewas_mod.phewas("30740", min_cases=50, ctx=ctx)

    assert result["biomarker"] == "Glucose"
    assert result["field_id"] == "30740"
    assert result["n_diseases_tested"] == 2
    assert result["top_associations"][0]["code"] == "E11"
    assert result["figure"].endswith("phewas_30740.png")
    assert len(ctx.state.figures) == 2

    none = phewas_mod.phewas("30740", min_cases=250, ctx=ctx)
    assert none["error"] == "No diseases with >= 250 cases found"
    plt.close("all")


class FakeSurvivalDM:
    def query(self, sql, params=None):
        if params == ["E11%"]:
            return pd.DataFrame({"eid": [1, 2, 3, 4]})
        if sql == "SELECT * FROM deaths":
            return pd.DataFrame(
                {
                    "eid": [1, 5],
                    "cause_icd10": ["I25", "C34"],
                    "assessment_date": ["2014-01-01", "2018-01-01"],
                }
            )
        if '"53-0.0"' in sql:
            return pd.DataFrame(
                {
                    "eid": list(range(1, 13)),
                    "age": np.linspace(50, 70, 12),
                    "assessment_date": ["2010-01-01"] * 12,
                }
            )
        raise AssertionError(sql)


def test_survival_uses_lifelines_and_actual_death_dates(tmp_path, monkeypatch):
    ctx = analysis_ctx(tmp_path, dm=FakeSurvivalDM())
    monkeypatch.setattr(survival_mod, "save_figure", fake_save_figure)
    monkeypatch.setattr(survival_mod, "log_rank_test", lambda *args, **kwargs: {"p_value": 0.0123})

    class FakeKaplanMeierFitter:
        def fit(self, durations, event_observed=None, label=None):
            self.durations = list(durations)
            self.events = list(event_observed)
            self.label = label
            self.median_survival_time_ = 9.5 if "Cases" in label else np.inf
            return self

        def plot_survival_function(self, ax, **kwargs):
            ax.plot([0, 1], [1, 0.95], label=self.label)
            return ax

    monkeypatch.setitem(sys.modules, "lifelines", SimpleNamespace(KaplanMeierFitter=FakeKaplanMeierFitter))

    result = survival_mod.survival("E11", ctx=ctx)

    assert result["n_cases"] == 4
    assert result["n_controls"] == 8
    assert result["case_deaths"] == 1
    assert result["control_deaths"] == 1
    assert result["log_rank_p"] == 0.0123
    assert result["case_median_survival"] == 9.5
    assert result["control_median_survival"] is None
    assert result["figure"].endswith("survival_E11.png")
    plt.close("all")


def test_survival_fallbacks_for_missing_tables_and_lifelines(tmp_path, monkeypatch):
    class FallbackDM(FakeSurvivalDM):
        def query(self, sql, params=None):
            if params == ["E11%"]:
                return pd.DataFrame({"eid": [1, 2]})
            if sql == "SELECT * FROM deaths":
                raise RuntimeError("no deaths table")
            if '"53-0.0"' in sql:
                raise RuntimeError("assessment date unavailable")
            return pd.DataFrame({"eid": list(range(1, 8)), "age": np.linspace(50, 60, 7)})

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lifelines":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = survival_mod.survival("E11", ctx=analysis_ctx(tmp_path, dm=FallbackDM()))

    assert result == {"error": "lifelines not installed. pip install lifelines"}


def test_survival_uses_jitter_when_death_dates_are_unavailable(tmp_path, monkeypatch):
    class JitterDM:
        def query(self, sql, params=None):
            if params == ["E11%"]:
                return pd.DataFrame({"eid": [1, 2]})
            if sql == "SELECT * FROM deaths":
                return pd.DataFrame({"eid": [1, 3], "cause_icd10": ["I25", "C34"]})
            if '"53-0.0"' in sql:
                return pd.DataFrame({"eid": list(range(1, 7)), "age": np.linspace(50, 60, 6)})
            raise AssertionError(sql)

    class FakeKaplanMeierFitter:
        def fit(self, durations, event_observed=None, label=None):
            self.durations = list(durations)
            self.events = list(event_observed)
            self.label = label
            self.median_survival_time_ = np.inf
            return self

        def plot_survival_function(self, ax, **kwargs):
            ax.plot([0, 1], [1, 0.9], label=self.label)
            return ax

    monkeypatch.setitem(sys.modules, "lifelines", SimpleNamespace(KaplanMeierFitter=FakeKaplanMeierFitter))
    monkeypatch.setattr(survival_mod.np.random, "beta", lambda *args, **kwargs: np.array([0.5, 0.25]))
    monkeypatch.setattr(survival_mod, "save_figure", fake_save_figure)
    monkeypatch.setattr(survival_mod, "log_rank_test", lambda *args, **kwargs: {})

    result = survival_mod.survival("E11", ctx=analysis_ctx(tmp_path, dm=JitterDM()))

    assert result["case_deaths"] == 1
    assert result["control_deaths"] == 1
    assert result["log_rank_p"] is None
    assert result["case_median_survival"] is None
    plt.close("all")


def test_survival_death_date_branch_handles_no_matching_deceased_subjects(tmp_path, monkeypatch):
    class NoMatchingDeathsDM(FakeSurvivalDM):
        def query(self, sql, params=None):
            if params == ["E11%"]:
                return pd.DataFrame({"eid": [1, 2]})
            if sql == "SELECT * FROM deaths":
                return pd.DataFrame({"eid": [999], "assessment_date": ["2020-01-01"]})
            if '"53-0.0"' in sql:
                return pd.DataFrame({
                    "eid": list(range(1, 7)),
                    "age": np.linspace(50, 60, 6),
                    "assessment_date": ["2010-01-01"] * 6,
                })
            raise AssertionError(sql)

    class FakeKaplanMeierFitter:
        def fit(self, durations, event_observed=None, label=None):
            self.durations = list(durations)
            self.events = list(event_observed)
            self.label = label
            self.median_survival_time_ = np.inf
            return self

        def plot_survival_function(self, ax, **kwargs):
            ax.plot([0, 1], [1, 0.95], label=self.label)
            return ax

    monkeypatch.setitem(sys.modules, "lifelines", SimpleNamespace(KaplanMeierFitter=FakeKaplanMeierFitter))
    monkeypatch.setattr(survival_mod, "save_figure", fake_save_figure)
    monkeypatch.setattr(survival_mod, "log_rank_test", lambda *args, **kwargs: {"p_value": 1.0})

    result = survival_mod.survival("E11", ctx=analysis_ctx(tmp_path, dm=NoMatchingDeathsDM()))

    assert result["case_deaths"] == 0
    assert result["control_deaths"] == 0
    plt.close("all")


def test_embedding_errors_tsne_success_and_umap_missing(tmp_path, monkeypatch):
    ctx = analysis_ctx(tmp_path)
    assert "No feature matrix" in embedding_mod.embedding(ctx=ctx)["error"]

    ctx.state.feature_matrix = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 20),
            "b": np.linspace(1, 0, 20),
        }
    )
    ctx.state.labels = pd.Series([0, 1] * 10)
    monkeypatch.setattr(embedding_mod, "save_figure", fake_save_figure)

    class FakeTSNE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, X):
            return np.column_stack([np.arange(len(X)), np.arange(len(X)) * -1])

    monkeypatch.setattr("sklearn.manifold.TSNE", FakeTSNE)
    tsne = embedding_mod.embedding(method="tsne", sample_size=10, ctx=ctx)

    assert tsne["method"] == "tsne"
    assert tsne["n_subjects"] == 10
    assert tsne["n_features"] == 2
    assert tsne["n_cases"] + tsne["n_controls"] == 10
    assert tsne["figure"].endswith("embedding_tsne.png")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    missing = embedding_mod.embedding(method="umap", sample_size=10, ctx=ctx)
    assert missing["error"] == "umap-learn not installed. pip install umap-learn"
    plt.close("all")


def test_embedding_umap_success_with_fake_backend(tmp_path, monkeypatch):
    ctx = analysis_ctx(tmp_path)
    ctx.state.feature_matrix = pd.DataFrame({
        "a": np.linspace(0, 1, 12),
        "b": np.linspace(1, 0, 12),
    })
    ctx.state.labels = pd.Series([0, 1] * 6)
    monkeypatch.setattr(embedding_mod, "save_figure", fake_save_figure)

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, X):
            return np.column_stack([np.arange(len(X)), np.arange(len(X)) * -1])

    monkeypatch.setattr(builtins, "__import__", __import__)
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))

    result = embedding_mod.embedding(method="umap", sample_size=6, ctx=ctx)

    assert result["method"] == "umap"
    assert result["n_subjects"] == 6
    assert result["figure"].endswith("embedding_umap.png")
    plt.close("all")


def gwas_cohort(n_cases=60, n_controls=240):
    labels = np.array([1] * n_cases + [0] * n_controls)
    return pd.DataFrame({"eid": np.arange(1, len(labels) + 1), "label": labels})


class FakeGwasDM:
    def get_field(self, fid):
        if fid == "missing":
            return pd.DataFrame()
        if fid == "bad":
            raise RuntimeError("field read failed")
        values = np.r_[np.linspace(5, 8, 60), np.linspace(1, 3, 240)]
        return pd.DataFrame({"eid": np.arange(1, 301), f"{fid}-0.0": values})


def test_gwas_proxy_error_paths_and_fdr_success(tmp_path, monkeypatch):
    ctx = analysis_ctx(tmp_path, dm=FakeGwasDM())
    monkeypatch.setattr("biobank_agent.data.features.BIOMARKER_GROUPS", {"blood": ["30740", "missing", "bad"]})
    monkeypatch.setattr("biobank_agent.data.cohort.build_cohort", lambda icd10_code, dm: gwas_cohort())
    monkeypatch.setattr(gwas_mod, "_plot_manhattan", lambda results, icd10_code, correction, ctx: [tmp_path / "gwas.png"])

    result = gwas_mod.gwas_proxy("E11", correction="fdr", min_effect_size=0.1, ctx=ctx)

    assert result["n_cases"] == 60
    assert result["n_controls"] == 240
    assert result["n_fields_tested"] == 1
    assert result["n_significant"] == 1
    assert result["top_associations"][0]["field_name"] == "Glucose"
    assert result["figures"] == [str(tmp_path / "gwas.png")]
    assert ctx.state.figures == [tmp_path / "gwas.png"]

    monkeypatch.setattr("biobank_agent.data.cohort.build_cohort", lambda icd10_code, dm: gwas_cohort(n_cases=10))
    assert "Too few cases" in gwas_mod.gwas_proxy("E11", ctx=ctx)["error"]

    monkeypatch.setattr("biobank_agent.data.cohort.build_cohort", lambda icd10_code, dm: (_ for _ in ()).throw(RuntimeError("bad cohort")))
    assert "Failed to build cohort" in gwas_mod.gwas_proxy("E11", ctx=ctx)["error"]


def test_gwas_proxy_bonferroni_and_no_testable_fields(tmp_path, monkeypatch):
    ctx = analysis_ctx(tmp_path, dm=FakeGwasDM())
    monkeypatch.setattr("biobank_agent.data.cohort.build_cohort", lambda icd10_code, dm: gwas_cohort())
    monkeypatch.setattr(gwas_mod, "_plot_manhattan", lambda *args, **kwargs: [tmp_path / "bonf.png"])

    monkeypatch.setattr("biobank_agent.data.features.BIOMARKER_GROUPS", {"blood": ["30740", "30750"]})
    bonf = gwas_mod.gwas_proxy("E11", correction="bonferroni", min_effect_size=0.1, ctx=ctx)
    assert bonf["correction"] == "bonferroni"
    assert bonf["n_fields_tested"] == 2
    assert all(hit["p_corrected"] <= 1.0 for hit in bonf["top_associations"])

    monkeypatch.setattr("biobank_agent.data.features.BIOMARKER_GROUPS", {"blood": ["missing", "bad"]})
    no_fields = gwas_mod.gwas_proxy("E11", ctx=ctx)
    assert no_fields["error"] == "No fields could be tested. Check data availability."


def test_gwas_proxy_skips_underpowered_fields_and_handles_zero_variance(tmp_path, monkeypatch):
    class SparseGwasDM:
        def get_field(self, fid):
            if fid == "small":
                return pd.DataFrame({"eid": list(range(1, 11)) + list(range(61, 71)), "small-0.0": [1.0] * 20})
            return pd.DataFrame({"eid": np.arange(1, 301), "flat-0.0": [5.0] * 300})

    ctx = analysis_ctx(tmp_path, dm=SparseGwasDM())
    delattr(ctx, "catalog")
    monkeypatch.setattr("biobank_agent.data.features.BIOMARKER_GROUPS", {"blood": ["small", "flat"]})
    monkeypatch.setattr("biobank_agent.data.cohort.build_cohort", lambda icd10_code, dm: gwas_cohort())
    monkeypatch.setattr(gwas_mod, "_plot_manhattan", lambda *args, **kwargs: [tmp_path / "flat.png"])

    result = gwas_mod.gwas_proxy("E11", correction="fdr", min_effect_size=0.0, ctx=ctx)

    assert result["n_fields_tested"] == 1
    assert result["all_results_count"] == 1
    assert result["n_significant"] == 0

    ctx_with_missing_catalog_entry = analysis_ctx(tmp_path, dm=SparseGwasDM())
    monkeypatch.setattr(gwas_mod, "_plot_manhattan", lambda *args, **kwargs: [])
    second = gwas_mod.gwas_proxy("E11", correction="fdr", min_effect_size=0.0, ctx=ctx_with_missing_catalog_entry)
    assert second["n_fields_tested"] == 1


def test_gwas_proxy_manhattan_plot_bonferroni_and_fdr(tmp_path):
    ctx = analysis_ctx(tmp_path, dm=FakeGwasDM())
    results = [
        {
            "group": "blood_biochemistry",
            "p_value": 1e-9,
            "significant": True,
            "abs_effect": 0.3,
            "field_name": "VeryLongBiomarkerNameForLabel",
        },
        {
            "group": "urine_assay",
            "p_value": 0.2,
            "significant": False,
            "abs_effect": 0.05,
            "field_name": "Weak",
        },
    ]

    bonf = gwas_mod._plot_manhattan(results, "E11", "bonferroni", ctx)
    fdr = gwas_mod._plot_manhattan(results, "E11", "fdr", ctx)

    assert all(path.exists() for path in [*bonf, *fdr])


def test_gwas_proxy_manhattan_accepts_array_axes(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    def fake_nature_figure(width="single", height_ratio=0.5):
        fig, ax = plt.subplots()
        return fig, np.array([ax])

    monkeypatch.setattr("biobank_agent.utils.plotting.nature_figure", fake_nature_figure)
    monkeypatch.setattr("biobank_agent.utils.plotting.save_figure", fake_save_figure)
    ctx = analysis_ctx(tmp_path, dm=FakeGwasDM())

    paths = gwas_mod._plot_manhattan(
        [
            {
                "group": "blood",
                "p_value": 0.5,
                "significant": False,
                "abs_effect": 0.0,
                "field_name": "Flat",
            }
        ],
        "E11",
        "fdr",
        ctx,
    )

    assert paths == [tmp_path / "gwas_proxy_E11.png", tmp_path / "gwas_proxy_E11.svg"]
    plt.close("all")
