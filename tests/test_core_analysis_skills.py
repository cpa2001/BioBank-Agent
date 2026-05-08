"""Focused tests for core analysis skills on compact synthetic contexts."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biobank_agent.data import features
from biobank_agent.skills import biomarker_dist as biomarker_mod
from biobank_agent.skills import cohort_summary as cohort_mod
from biobank_agent.skills import comorbidity as comorbidity_mod
from biobank_agent.skills import correlation as correlation_mod
from biobank_agent.skills import data_query as data_query_mod
from biobank_agent.skills import missing_data as missing_mod
from biobank_agent.skills import think as think_mod


class SkillState:
    def __init__(self):
        self.cohorts = {}
        self.figures = []

    def context_summary(self):
        return "prior analyses"


class SkillCtx:
    def __init__(self, tmp_path, dm=None, catalog=None):
        self.report_dir = tmp_path
        self.state = SkillState()
        self.dm = dm
        self.catalog = catalog
        self.settings = SimpleNamespace(
            subject_id_col="eid",
            diagnoses_code_col="diag_icd10",
            enable_tot=False,
            llm_base_url="http://llm",
            llm_api_key="key",
            llm_model="model",
        )


def fake_save_figure(fig, name, report_dir, formats=("png", "svg")):
    suffixes = formats if isinstance(formats, tuple) else ("png",)
    return [report_dir / f"{name}.{suffix}" for suffix in suffixes]


def make_cohort(n_cases=12, n_controls=48, include_demographics=True):
    labels = np.array([1] * n_cases + [0] * n_controls)
    n = len(labels)
    df = pd.DataFrame(
        {
            "eid": np.arange(n) + 1,
            "label": labels,
            "30740-0.0": np.r_[np.linspace(7, 9, n_cases), np.linspace(4, 6, n_controls)],
            "30750-0.0": np.r_[np.linspace(60, 75, n_cases), np.linspace(35, 46, n_controls)],
        }
    )
    if include_demographics:
        df["31-0.0"] = [1, 0] * (n // 2)
        df["21003-0.0"] = np.r_[np.linspace(60, 70, n_cases), np.linspace(45, 65, n_controls)]
    return df


def test_biomarker_dist_builds_and_reuses_cohort(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path)
    cohort = make_cohort()
    calls = []

    def fake_build_cohort(dm, icd10_code, controls_ratio=4):
        calls.append((icd10_code, controls_ratio))
        return cohort

    monkeypatch.setattr(biomarker_mod, "build_cohort", fake_build_cohort)
    monkeypatch.setattr(biomarker_mod, "save_figure", fake_save_figure)

    result = biomarker_mod.biomarker_dist("E11", biomarkers="30740,missing", ctx=ctx)
    reused = biomarker_mod.biomarker_dist("E11", biomarkers="30750", ctx=ctx)

    assert calls == [("E11", 4)]
    assert result["disease"] == "Type 2 diabetes mellitus"
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["field_id"] == "30740"
    assert result["figure"].endswith("biomarker_dist_E11.png")
    assert reused["comparisons"][0]["field_id"] == "30750"
    assert len(ctx.state.figures) == 4
    plt.close("all")


def test_biomarker_dist_skips_small_groups_without_plot(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path)
    monkeypatch.setattr(biomarker_mod, "build_cohort", lambda *args, **kwargs: make_cohort(n_cases=2, n_controls=8))

    result = biomarker_mod.biomarker_dist("E11", biomarkers="30740", ctx=ctx)

    assert result["comparisons"] == []
    assert result["figure"] is None


def test_biomarker_dist_default_markers_hide_unused_axes(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path)
    cohort = make_cohort()
    cohort["30690-0.0"] = np.r_[np.linspace(1, 2, 12), np.linspace(2, 3, 48)]
    cohort["30870-0.0"] = np.r_[np.linspace(3, 4, 12), np.linspace(4, 5, 48)]
    monkeypatch.setattr(biomarker_mod, "build_cohort", lambda *args, **kwargs: cohort)
    monkeypatch.setattr(biomarker_mod, "save_figure", fake_save_figure)

    result = biomarker_mod.biomarker_dist("E11", ctx=ctx)

    assert [row["field_id"] for row in result["comparisons"]] == ["30740", "30750", "30690", "30870"]
    assert result["figure"].endswith("biomarker_dist_E11.png")
    plt.close("all")


def test_cohort_summary_demographics_plot_and_no_age_path(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path)
    monkeypatch.setattr(cohort_mod, "build_cohort", lambda *args, **kwargs: make_cohort())
    monkeypatch.setattr(cohort_mod, "save_figure", fake_save_figure)

    result = cohort_mod.cohort_summary("E11", controls_ratio=2, ctx=ctx)

    assert result["n_cases"] == 12
    assert result["n_controls"] == 48
    assert result["sex"]["cases_male_pct"] == pytest.approx(50.0)
    assert "cases_age" in result["age"]
    assert result["figure"].endswith("cohort_E11_age.png")

    cached = cohort_mod.cohort_summary("E11", controls_ratio=2, ctx=ctx)
    assert cached["n_cases"] == 12

    ctx_no_age = SkillCtx(tmp_path)
    monkeypatch.setattr(
        cohort_mod,
        "build_cohort",
        lambda *args, **kwargs: make_cohort(include_demographics=False),
    )
    no_age = cohort_mod.cohort_summary("E11", ctx=ctx_no_age)

    assert no_age["sex"] == {}
    assert no_age["age"] == {}
    assert no_age["figure"] is None
    plt.close("all")


class FakeComorbidityDM:
    def query(self, sql, params=None):
        if "SELECT DISTINCT eid" in sql and "FROM diagnoses" in sql and "LEFT" not in sql:
            return pd.DataFrame({"eid": list(range(1, 201))})
        if "LEFT(diag_icd10, 3) AS code" in sql:
            return pd.DataFrame({"code": ["I10", "E78"], "n_co": [150, 120]})
        if "LIKE 'I10%'" in sql:
            return pd.DataFrame({"n": [250]})
        if "LIKE 'E78%'" in sql:
            return pd.DataFrame({"n": [500]})
        raise AssertionError(sql)

    def count_subjects(self):
        return 1000


def test_comorbidity_computes_odds_ratios_and_saves_plot(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path, dm=FakeComorbidityDM())
    monkeypatch.setattr(comorbidity_mod, "save_figure", fake_save_figure)

    result = comorbidity_mod.comorbidity("E11", top_n=2, ctx=ctx)

    assert result["n_target_patients"] == 200
    assert [r["code"] for r in result["comorbidities"]] == ["I10", "E78"]
    assert result["comorbidities"][0]["odds_ratio"] == 21.0
    assert result["figure"].endswith("comorbidity_E11.png")
    assert len(ctx.state.figures) == 2
    plt.close("all")


def test_comorbidity_skips_invalid_odds_ratio_rows(tmp_path, monkeypatch):
    class InvalidComorbidityDM:
        def count_subjects(self):
            return 100

        def query(self, sql, params=None):
            if "GROUP BY code" in sql:
                return pd.DataFrame({"code": ["I10"], "n_co": [10]})
            if "COUNT(DISTINCT" in sql:
                return pd.DataFrame({"n": [10]})
            if "SELECT DISTINCT" in sql:
                return pd.DataFrame({"eid": list(range(10))})
            raise AssertionError(sql)

    ctx = SkillCtx(tmp_path, dm=InvalidComorbidityDM())
    monkeypatch.setattr(comorbidity_mod, "save_figure", fake_save_figure)

    result = comorbidity_mod.comorbidity("E11", ctx=ctx)

    assert result["comorbidities"] == []
    assert result["figure"].endswith("comorbidity_E11.png")
    plt.close("all")


class FakeCorrelationDM:
    def query(self, sql, params=None):
        assert "USING SAMPLE 50000" in sql
        return pd.DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": [2, 4, 6, 8],
                "C": [4, 3, 2, 1],
            }
        )


class FakeWeakCorrelationDM:
    def query(self, sql, params=None):
        return pd.DataFrame(
            {
                "A": [0, 1, 2, 3, 4, 5, 6, 7],
                "B": [0, 6, 5, 3, 2, 1, 7, 4],
                "C": [3, 2, 1, 4, 6, 0, 5, 7],
            }
        )


def test_correlation_uses_requested_group_and_reports_pairs(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path, dm=FakeCorrelationDM())
    monkeypatch.setattr(correlation_mod, "BLOOD_BIOCHEMISTRY", {"1": "A", "2": "B", "3": "C"})
    monkeypatch.setattr(correlation_mod, "apply_nature_style", lambda: None)
    monkeypatch.setattr(correlation_mod, "save_figure", fake_save_figure)

    class FakeCluster:
        def __init__(self):
            self.fig, _ = plt.subplots()
            self.ax_heatmap = self.fig.axes[0]

    monkeypatch.setattr(correlation_mod.sns, "clustermap", lambda *args, **kwargs: FakeCluster())

    result = correlation_mod.correlation("unknown-group", ctx=ctx)

    assert result["group"] == "unknown-group"
    assert result["n_features"] == 3
    assert result["figures"] == [
        str(tmp_path / "correlation_unknown-group.svg"),
        str(tmp_path / "correlation_unknown-group.pdf"),
    ]
    assert result["top_correlations"][0]["r"] == 1.0
    plt.close("all")


def test_correlation_reports_no_strong_pairs(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path, dm=FakeWeakCorrelationDM())
    monkeypatch.setattr(correlation_mod, "BLOOD_BIOCHEMISTRY", {"1": "A", "2": "B", "3": "C"})
    monkeypatch.setattr(correlation_mod, "apply_nature_style", lambda: None)
    monkeypatch.setattr(correlation_mod, "save_figure", fake_save_figure)

    class FakeCluster:
        def __init__(self):
            self.fig, _ = plt.subplots()
            self.ax_heatmap = self.fig.axes[0]

    monkeypatch.setattr(correlation_mod.sns, "clustermap", lambda *args, **kwargs: FakeCluster())

    result = correlation_mod.correlation("biochemistry", ctx=ctx)

    assert result["top_correlations"] == []
    plt.close("all")


class FakeMissingDM:
    def query(self, sql, params=None):
        assert "USING SAMPLE 4" in sql
        return pd.DataFrame(
            {
                "Glucose": [1.0, None, 3.0, None],
                "HbA1c": [5.0, 6.0, 7.0, 8.0],
                "BMI": [None, None, None, 1.0],
            }
        )


def test_missing_data_categorizes_features_and_records_figure(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path, dm=FakeMissingDM())
    monkeypatch.setattr(missing_mod, "ALL_BIOMARKERS", {"30740": "Glucose", "30750": "HbA1c", "21001": "BMI"})
    monkeypatch.setattr(missing_mod, "save_figure", fake_save_figure)

    result = missing_mod.missing_data(sample_size=4, ctx=ctx)

    assert result["n_features"] == 3
    assert result["overall_missing_pct"] == 41.67
    assert result["complete_features"] == 1
    assert result["high_missing"] == 2
    assert result["top_missing"][0] == {"feature": "BMI", "missing_pct": 75.0}
    assert result["figure"].endswith("missing_data.png")
    plt.close("all")


def test_field_search_numeric_keyword_and_missing_paths(tmp_path):
    class FakeCatalog:
        def field_info(self, field_id):
            return {
                "30740": {
                    "title": "Glucose",
                    "category_id": "biochem",
                    "value_type": "Continuous",
                    "units": "mmol/L",
                }
            }.get(field_id)

        def category_name(self, category_id):
            return {"biochem": "Blood biochemistry"}.get(category_id, "Unknown")

        def search(self, query, limit=20):
            assert query == "glucose"
            assert limit == 1
            return [{"field_id": "30740", "title": "Glucose", "category_id": "biochem"}]

    class FakeDM:
        def field_source(self, field_id):
            return "parquet"

    ctx = SkillCtx(tmp_path, dm=FakeDM(), catalog=FakeCatalog())

    by_id = data_query_mod.field_search("30740", ctx=ctx)
    missing = data_query_mod.field_search("99999", ctx=ctx)
    keyword = data_query_mod.field_search("glucose", limit=1, ctx=ctx)
    no_query = data_query_mod.field_search(ctx=ctx)
    no_catalog = data_query_mod.field_search("glucose", ctx=SimpleNamespace())

    no_dm_ctx = SkillCtx(tmp_path, catalog=FakeCatalog())
    no_dm_ctx.dm = None
    unknown_source = data_query_mod.field_search("glucose", limit=1, ctx=no_dm_ctx)

    assert by_id["results"][0]["title"] == "Glucose"
    assert missing["total"] == 0
    assert keyword["results"][0]["data_source"] == "parquet"
    assert no_query["error"] == "Missing query"
    assert no_catalog["error"] == "Field catalogue unavailable"
    assert unknown_source["results"][0]["data_source"] == "unknown"


def test_think_simple_tot_success_and_tot_fallback(tmp_path, monkeypatch):
    ctx = SkillCtx(tmp_path)
    assert think_mod.think("linear thought", ctx=ctx) == {"acknowledged": True}
    assert think_mod.think(thought="alias thought", limit=2, ctx=ctx) == {"acknowledged": True}
    assert think_mod.think(limit=2, query="query thought", ctx=ctx) == {"acknowledged": True}

    ctx.settings.enable_tot = True
    monkeypatch.setattr("biobank_agent.reasoning.is_branching_question", lambda reasoning: True)

    class FakeLLMClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTree:
        def __init__(self, llm, max_branches=3):
            self.llm = llm
            self.max_branches = max_branches

        def explore(self, reasoning, context=""):
            path = SimpleNamespace(
                approach="cohort first",
                score=0.9,
                steps=["build cohort"] * 6,
                rationale="because it constrains denominators" * 20,
            )
            return SimpleNamespace(paths=[path], recommended=path, consensus="Use cohort-first analysis.")

    monkeypatch.setattr("biobank_agent.llm.LLMClient", FakeLLMClient)
    monkeypatch.setattr("biobank_agent.reasoning.ThoughtTree", FakeTree)

    tot = think_mod.think("Which analysis approach should I use?", ctx=ctx)
    assert tot["tree_of_thought"] is True
    assert tot["paths"][0]["steps"] == ["build cohort"] * 5
    assert len(tot["paths"][0]["rationale"]) == 200
    assert tot["recommended"] == "cohort first"

    class RaisingTree(FakeTree):
        def explore(self, reasoning, context=""):
            raise RuntimeError("llm unavailable")

    monkeypatch.setattr("biobank_agent.reasoning.ThoughtTree", RaisingTree)
    assert think_mod.think("Which approach now?", ctx=ctx) == {"acknowledged": True}
    assert think_mod.think("no settings", ctx=SimpleNamespace()) == {"acknowledged": True}

    non_branching = SkillCtx(tmp_path)
    non_branching.settings.enable_tot = True
    monkeypatch.setattr("biobank_agent.reasoning.is_branching_question", lambda reasoning: False)
    assert think_mod.think("Summarize this analysis", ctx=non_branching) == {"acknowledged": True}


def test_feature_group_lookup_and_rename_columns():
    df = pd.DataFrame(columns=["eid", "label", "30740-0.0", "999-0.0", "31-0.0"])
    catalog = SimpleNamespace(field_name=lambda fid: {"999": "Custom field"}.get(fid))

    assert features.get_feature_group("biochemistry") is features.BLOOD_BIOCHEMISTRY
    assert features.get_feature_group("does-not-exist") is features.ALL_BIOMARKERS
    assert features.rename_columns(df, catalog=catalog) == {
        "30740-0.0": "Glucose",
        "999-0.0": "Custom field",
        "31-0.0": "Sex",
    }
    assert features.rename_columns(pd.DataFrame(columns=["unknown-0.0"])) == {}
