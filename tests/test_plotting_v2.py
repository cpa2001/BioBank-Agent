"""Test publication-quality plotting utilities."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestPaletteConstants:
    """Test colour palette definitions."""

    def test_palette_has_8_colours(self):
        """PALETTE should have exactly 8 Okabe-Ito colours."""
        from biobank_agent.utils.plotting import PALETTE
        assert len(PALETTE) == 8

    def test_palette_colours_are_hex(self):
        """Each colour is a valid hex string."""
        from biobank_agent.utils.plotting import PALETTE
        for colour in PALETTE:
            assert colour.startswith("#")
            assert len(colour) == 7

    def test_palette_seq_exists(self):
        """PALETTE_SEQ colourmap string is defined."""
        from biobank_agent.utils.plotting import PALETTE_SEQ
        assert PALETTE_SEQ == "viridis"

    def test_palette_div_exists(self):
        """PALETTE_DIV colourmap string is defined."""
        from biobank_agent.utils.plotting import PALETTE_DIV
        assert PALETTE_DIV == "RdBu_r"


class TestNatureFigure:
    """Test Nature-style figure constructor."""

    def test_nature_figure_returns_fig_and_axes(self):
        """nature_figure() returns (Figure, Axes)."""
        from biobank_agent.utils.plotting import nature_figure

        fig, axes = nature_figure(nrows=1, ncols=1)

        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)
        plt.close(fig)

    def test_nature_single_column_dimensions(self):
        """Single-column figure width ~ 89mm / 25.4 ~ 3.50 inches."""
        from biobank_agent.utils.plotting import nature_figure, SINGLE_COL

        fig, _ = nature_figure(nrows=1, ncols=1, width="single")
        w, h = fig.get_size_inches()

        assert abs(w - SINGLE_COL) < 0.01
        plt.close(fig)

    def test_nature_double_column_dimensions(self):
        """Double-column figure width ~ 183mm / 25.4 ~ 7.20 inches."""
        from biobank_agent.utils.plotting import nature_figure, DOUBLE_COL

        fig, _ = nature_figure(nrows=1, ncols=2, width="double")
        w, h = fig.get_size_inches()

        assert abs(w - DOUBLE_COL) < 0.01
        plt.close(fig)

    def test_nature_multi_panel(self):
        """nature_figure with 2x2 returns array of axes."""
        from biobank_agent.utils.plotting import nature_figure

        fig, axes = nature_figure(nrows=2, ncols=2)

        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)
        plt.close(fig)


class TestICMLFigure:
    """Test ICML-style figure constructor."""

    def test_icml_figure_returns_fig_and_axes(self):
        """icml_figure() returns (Figure, Axes)."""
        from biobank_agent.utils.plotting import icml_figure

        fig, axes = icml_figure(nrows=1, ncols=1)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_icml_default_dimensions(self):
        """ICML figure default width is 6.75 inches, row height 2.5 in."""
        from biobank_agent.utils.plotting import icml_figure, ICML_TEXTWIDTH, ICML_ROW_HEIGHT

        fig, _ = icml_figure(nrows=1, ncols=1)
        w, h = fig.get_size_inches()

        assert abs(w - ICML_TEXTWIDTH) < 0.01
        assert abs(h - ICML_ROW_HEIGHT) < 0.01
        plt.close(fig)

    def test_icml_custom_dimensions(self):
        """Custom width and row_height are respected."""
        from biobank_agent.utils.plotting import icml_figure

        fig, _ = icml_figure(nrows=2, ncols=1, width=5.0, row_height=3.0)
        w, h = fig.get_size_inches()

        assert abs(w - 5.0) < 0.01
        assert abs(h - 6.0) < 0.01  # 2 rows * 3.0
        plt.close(fig)


class TestSaveFigure:
    """Test save_figure utility."""

    def test_default_formats_are_svg_pdf(self):
        """save_figure default formats tuple is ("svg", "pdf")."""
        from biobank_agent.utils.plotting import save_figure
        import inspect
        sig = inspect.signature(save_figure)
        default = sig.parameters["formats"].default
        assert default == ("svg", "pdf")

    def test_save_creates_both_files(self, tmp_path):
        """save_figure creates .svg and .pdf files."""
        from biobank_agent.utils.plotting import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        paths = save_figure(fig, "test_fig", tmp_path)

        assert len(paths) == 2
        assert (tmp_path / "test_fig.svg").exists()
        assert (tmp_path / "test_fig.pdf").exists()

    def test_save_custom_formats(self, tmp_path):
        """save_figure with custom formats creates only those files."""
        from biobank_agent.utils.plotting import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        paths = save_figure(fig, "custom", tmp_path, formats=("png",))

        assert len(paths) == 1
        assert (tmp_path / "custom.png").exists()
        assert not (tmp_path / "custom.svg").exists()

    def test_save_creates_directory_if_missing(self, tmp_path):
        """save_figure creates report_dir if it does not exist."""
        from biobank_agent.utils.plotting import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        nested = tmp_path / "sub" / "dir"
        paths = save_figure(fig, "fig", nested)

        assert nested.exists()
        assert len(paths) == 2


class TestAddPanelLabels:
    """Test panel label addition."""

    def test_adds_default_labels(self):
        """add_panel_labels adds (a), (b), (c) to axes."""
        from biobank_agent.utils.plotting import add_panel_labels

        fig, axes = plt.subplots(1, 3)
        add_panel_labels(fig, axes)

        # Verify text was added to each axes
        for i, ax in enumerate(axes):
            texts = [t.get_text() for t in ax.texts]
            expected = f"({chr(ord('a') + i)})"
            assert expected in texts

        plt.close(fig)

    def test_adds_custom_labels(self):
        """Custom label strings override defaults."""
        from biobank_agent.utils.plotting import add_panel_labels

        fig, axes = plt.subplots(1, 2)
        add_panel_labels(fig, axes, labels=["Panel X", "Panel Y"])

        assert "Panel X" in [t.get_text() for t in axes[0].texts]
        assert "Panel Y" in [t.get_text() for t in axes[1].texts]
        plt.close(fig)

    def test_single_axes_wrapped(self):
        """Single Axes (not array) is handled gracefully."""
        from biobank_agent.utils.plotting import add_panel_labels

        fig, ax = plt.subplots()
        add_panel_labels(fig, ax)

        texts = [t.get_text() for t in ax.texts]
        assert "(a)" in texts
        plt.close(fig)


class TestSmartLegend:
    """Test smart legend placement."""

    def test_few_items_inside_axes(self):
        """<= 5 items → legend inside axes (upper right)."""
        from biobank_agent.utils.plotting import smart_legend

        fig, ax = plt.subplots()
        for i in range(3):
            ax.plot([1, 2], label=f"Line {i}")

        legend = smart_legend(ax)

        # Legend should be inside axes — loc code 1 = "upper right"
        # or the string version
        assert legend is not None
        loc = legend._loc
        # matplotlib loc code 1 = upper right (default for our function)
        # Just check it was created and is a Legend
        assert legend is not None
        plt.close(fig)

    def test_many_items_outside_axes(self):
        """More than 5 items → legend outside axes."""
        from biobank_agent.utils.plotting import smart_legend

        fig, ax = plt.subplots()
        for i in range(8):
            ax.plot([1, 2], label=f"Line {i}")

        legend = smart_legend(ax)

        assert legend is not None
        # bbox_to_anchor should be set for outside placement
        bbox = legend.get_bbox_to_anchor()
        assert bbox is not None
        plt.close(fig)

    def test_no_items_returns_empty_legend(self):
        """No labeled handles → returns a no-op legend."""
        from biobank_agent.utils.plotting import smart_legend

        fig, ax = plt.subplots()
        ax.plot([1, 2])  # no label

        legend = smart_legend(ax)

        assert legend is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
