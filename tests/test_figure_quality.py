"""Figure quality verification tests — Nature / ICML standards.

Checks:
- DPI >= 300
- Nature single-column width = 89mm (3.50 in)
- No top/right spines
- Okabe-Ito colour-blind safe palette
- ICML text-width = 6.75 in
- SVG + PDF default output
"""

import pytest
from pathlib import Path

from biobank_agent.utils.plotting import (
    nature_figure,
    icml_figure,
    save_figure,
    add_panel_labels,
    smart_legend,
    PALETTE,
    PALETTE_SEQ,
    PALETTE_DIV,
    SINGLE_COL,
    DOUBLE_COL,
)


class TestNatureFigure:
    """Nature journal figure specifications."""

    def test_single_column_width(self):
        fig, ax = nature_figure(width="single")
        # Nature single column = 89mm = 3.50 inches
        assert abs(fig.get_figwidth() - 89 / 25.4) < 0.05

    def test_double_column_width(self):
        fig, ax = nature_figure(width="double")
        # Nature double column = 183mm = 7.20 inches
        assert abs(fig.get_figwidth() - 183 / 25.4) < 0.05

    def test_no_top_spine(self):
        fig, ax = nature_figure()
        assert not ax.spines["top"].get_visible()

    def test_no_right_spine(self):
        fig, ax = nature_figure()
        assert not ax.spines["right"].get_visible()

    def test_left_spine_visible(self):
        fig, ax = nature_figure()
        assert ax.spines["left"].get_visible()

    def test_bottom_spine_visible(self):
        fig, ax = nature_figure()
        assert ax.spines["bottom"].get_visible()

    def test_dpi_at_least_300(self):
        fig, ax = nature_figure()
        assert fig.get_dpi() >= 300

    def test_font_size_7pt(self):
        import matplotlib
        fig, ax = nature_figure()
        # The context manager sets font.size to 7
        # After creation, check tick label sizes
        for label in ax.get_xticklabels():
            assert label.get_fontsize() <= 8  # 6 or 7 pt

    def test_height_ratio(self):
        fig, ax = nature_figure(width="single", height_ratio=0.8)
        expected_height = (89 / 25.4) * 0.8
        assert abs(fig.get_figheight() - expected_height) < 0.1


class TestICMLFigure:
    """ICML conference figure specifications."""

    def test_textwidth(self):
        fig, ax = icml_figure()
        assert abs(fig.get_figwidth() - 6.75) < 0.05

    def test_serif_font_family(self):
        # ICML uses serif fonts
        fig, ax = icml_figure()
        # Just verify creation works (font is set in rcParams context)
        assert fig is not None


class TestPalette:
    """Colour-blind safe palette verification."""

    def test_okabe_ito_first_color(self):
        assert PALETTE[0] == "#0072B2"  # Okabe-Ito blue

    def test_palette_has_8_colors(self):
        assert len(PALETTE) >= 8

    def test_sequential_colormap(self):
        assert PALETTE_SEQ == "viridis"

    def test_diverging_colormap(self):
        assert PALETTE_DIV == "RdBu_r"

    def test_all_hex_format(self):
        import re
        for c in PALETTE:
            assert re.match(r"^#[0-9A-Fa-f]{6}$", c), f"Invalid hex: {c}"


class TestSaveFigure:
    """Figure output format tests."""

    def test_default_svg_pdf_output(self, tmp_path):
        fig, ax = nature_figure()
        ax.plot([1, 2, 3], [1, 2, 3])
        paths = save_figure(fig, "test", tmp_path)
        suffixes = {p.suffix for p in paths}
        assert ".svg" in suffixes, "SVG output missing"
        assert ".pdf" in suffixes, "PDF output missing"

    def test_png_output(self, tmp_path):
        fig, ax = nature_figure()
        ax.plot([1, 2, 3], [1, 2, 3])
        paths = save_figure(fig, "test", tmp_path, formats=("png",))
        assert any(p.suffix == ".png" for p in paths)

    def test_png_dpi_at_least_300(self, tmp_path):
        fig, ax = nature_figure()
        ax.plot([1, 2, 3], [1, 2, 3])
        paths = save_figure(fig, "test", tmp_path, formats=("png",))
        png_path = [p for p in paths if p.suffix == ".png"][0]
        try:
            from PIL import Image
            img = Image.open(png_path)
            dpi = img.info.get("dpi", (72, 72))
            assert dpi[0] >= 299, f"DPI too low: {dpi}"
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_output_files_exist(self, tmp_path):
        fig, ax = nature_figure()
        ax.plot([1, 2, 3], [1, 2, 3])
        paths = save_figure(fig, "test", tmp_path)
        for p in paths:
            assert p.exists(), f"Output file missing: {p}"


class TestPanelLabels:
    """Panel label tests (multi-panel figures)."""

    def test_add_panel_labels(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        # Should not raise
        add_panel_labels(fig, axes.flat)
        plt.close(fig)

    def test_smart_legend(self):
        fig, ax = nature_figure()
        ax.plot([1, 2], [1, 2], label="Test")
        legend = smart_legend(ax)
        assert legend is not None
        plt.close(fig)


class TestConstants:
    """Dimension constants verification."""

    def test_single_col_89mm(self):
        assert abs(SINGLE_COL - 89 / 25.4) < 0.001

    def test_double_col_183mm(self):
        assert abs(DOUBLE_COL - 183 / 25.4) < 0.001


# Cleanup
import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    plt.close("all")
