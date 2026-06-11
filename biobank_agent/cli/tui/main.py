"""Minimal Textual app shell for Biobank Agent."""

from __future__ import annotations

from typing import Any

from rich.console import Console


def textual_available() -> bool:
    try:
        import textual  # noqa: F401
    except Exception:
        return False
    return True


def run_tui(settings: Any | None = None) -> bool:
    """Run the Textual UI if available.

    Returns ``True`` when a Textual app was launched. If Textual is not
    installed, prints a clear message and returns ``False`` so the caller can
    choose a rich.live fallback.
    """
    if not textual_available():
        Console().print(
            "[yellow]Textual TUI is not installed.[/] "
            "Install with `pip install -e .[tui]` or run `biobank` for the rich CLI."
        )
        return False

    from .main_screen import BiobankTuiApp

    app = BiobankTuiApp(settings=settings)
    app.run()
    return True


__all__ = ["run_tui", "textual_available"]
