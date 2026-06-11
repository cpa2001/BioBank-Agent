from types import SimpleNamespace

from biobank_agent.skills.report import generate_report


def test_generate_report_materializes_figures_into_final_report_dir(tmp_path):
    source_dir = tmp_path / "earlier_run"
    source_dir.mkdir()
    svg = source_dir / "cohort_E11_age.svg"
    pdf = source_dir / "cohort_E11_age.pdf"
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4\n")

    report_dir = tmp_path / "final_report"
    state = SimpleNamespace(
        records=[],
        figures=[svg, pdf],
        cohorts={},
        models={},
        model_metadata={},
        provenances=[],
        executive_findings=[],
    )
    settings = SimpleNamespace(
        biobank_name="UK Biobank",
        biobank_caveats="",
        biobank_description="a prospective cohort",
    )
    ctx = SimpleNamespace(state=state, settings=settings, report_dir=report_dir)

    result = generate_report("Artifact Test", format="dual", ctx=ctx)

    assert result["format"] == "dual"
    assert result["n_figures"] == 1
    assert result["broken_figure_links"] == []
    assert "error" not in result
    assert (report_dir / "cohort_E11_age.svg").exists()
    assert (report_dir / "cohort_E11_age.pdf").exists()
    assert "![Figure 1](cohort_E11_age.svg)" in (report_dir / "report_technical.md").read_text(encoding="utf-8")
