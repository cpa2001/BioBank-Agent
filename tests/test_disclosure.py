"""Tests for progressive result disclosure formatting."""

from biobank_agent.disclosure import DisclosureLayer, DisclosedResult, ProgressivePresenter


def test_disclosure_layer_names_and_result_navigation():
    layers = [
        DisclosureLayer(1, "Headline", "one"),
        DisclosureLayer(2, "Summary", "two"),
        DisclosureLayer(5, "Custom", "five"),
    ]
    result = DisclosedResult("demo", layers=layers)

    assert layers[0].level_name == "Headline"
    assert layers[2].level_name == "Layer 5"
    assert result.get_current() == "one"
    assert "### Headline" in result.get_up_to(2)
    assert "### Summary" in result.get_up_to(2)
    assert "Custom" not in result.get_up_to(2)
    assert result.has_more() is True

    result.current_level = 4
    assert result.get_current() == "one"


def test_empty_disclosed_result_is_safe():
    result = DisclosedResult("empty")

    assert result.get_current() == ""
    assert result.get_up_to(4) == ""
    assert result.has_more() is False


def test_format_auto_routes_by_skill_and_keys():
    presenter = ProgressivePresenter(top_n_summary=2)

    assert presenter.format_auto("phewas", {"associations": []}).skill_name == "phewas"
    assert presenter.format_auto("train_model", {"auc": 0.75}).skill_name == "train_model"
    assert presenter.format_auto("x", {"hits": []}).skill_name == "gwas"
    assert presenter.format_auto("x", {"hazard_ratio": 1.2}).skill_name == "survival"
    assert presenter.format_auto("x", {"value": 1}).skill_name == "x"


def test_format_phewas_layers_and_truncates_summary():
    presenter = ProgressivePresenter(top_n_summary=2)
    result = presenter.format_phewas({
        "associations": [
            {"phenotype": "A", "p_value": 1e-9, "odds_ratio": 1.4, "beta": 0.2},
            {"phenotype": "B", "p_value": 0.04, "odds_ratio": 0.8, "beta": -0.1},
            {"phenotype": "C", "p_value": 0.02, "odds_ratio": 1.2},
        ]
    })

    assert result.total_records == 3
    assert len(result.layers) == 4
    assert "genome-wide significant" in result.layers[0].content
    assert result.layers[1].content.count("| ") >= 4
    assert "A" in result.layers[1].content
    assert "B" not in result.layers[1].content

    many = ProgressivePresenter().format_phewas({
        "associations": [
            {"phenotype": f"P{i}", "p_value": 0.1 + i * 1e-4}
            for i in range(205)
        ]
    })
    assert "... and 5 more" in many.layers[2].content


def test_format_phewas_falls_back_on_malformed_associations():
    result = ProgressivePresenter().format_phewas({"associations": {"bad": "shape"}})

    assert result.skill_name == "phewas"
    assert len(result.layers) == 2
    assert "completed" in result.layers[0].content


def test_format_model_result_includes_ci_metrics_and_features():
    result = ProgressivePresenter().format_model_result("train_model", {
        "auc": 0.8123,
        "auc_ci_low": 0.78,
        "auc_ci_high": 0.84,
        "accuracy": 0.7,
        "n_features": 12,
        "model_type": "xgboost",
        "feature_importance": [{"name": "HbA1c", "importance": 0.5}],
        "n_cases": 1000,
        "n_controls": 4000,
    })

    assert "AUC = 0.812" in result.layers[0].content
    assert "95% CI" in result.layers[0].content
    assert "HbA1c" in result.layers[1].content
    assert "Cross-validation" in result.layers[3].content

    tuple_features = ProgressivePresenter().format_model_result("train_model", {
        "auc": 0.7,
        "n_features": 2,
        "top_features": [("LDL", 0.4)],
    })
    assert "LDL" in tuple_features.layers[1].content

    sparse_features = ProgressivePresenter().format_model_result("train_model", {
        "auc": 0.7,
        "top_features": [("bad",)],
    })
    assert "bad" not in sparse_features.layers[1].content


def test_format_gwas_survival_and_generic_outputs():
    presenter = ProgressivePresenter(top_n_summary=1)
    gwas = presenter.format_gwas({
        "hits": [{"rsid": "rs1", "chr": "1", "pos": 123, "p_value": 1e-10, "gene": "GENE"}],
        "n_variants_tested": 1000,
    })
    survival = presenter.format_survival({
        "hazard_ratio": 1.5,
        "hr_ci_low": 1.2,
        "hr_ci_high": 1.8,
        "p_value": 0.001,
    })
    generic = presenter.format_generic("custom", {"alpha": 1, "nested": {"x": 2}})
    malformed_gwas = presenter.format_gwas({"hits": ["not-a-dict"], "n_variants_tested": 1})

    assert "rs1" in gwas.layers[1].content
    assert "not-a-dict" not in malformed_gwas.layers[1].content
    assert "Hazard Ratio" in survival.layers[0].content
    assert "alpha=1" in generic.layers[0].content
