"""Confirmation modal helpers for TUI approvals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - imported in environments with Textual
    from textual.containers import Container, Horizontal
    from textual.screen import ModalScreen
    from textual.widgets import Button, Static
except Exception:  # pragma: no cover - Textual optional in unit tests
    Container = Horizontal = Button = Static = None  # type: ignore[assignment]

    class ModalScreen:  # type: ignore[no-redef]
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

        def __init__(self, *_args, **_kwargs) -> None:
            pass


@dataclass(frozen=True)
class ConfirmationRequest:
    title: str
    body: str
    confirm_label: str = "Approve"
    cancel_label: str = "Cancel"


class ConfirmationModal(ModalScreen[bool]):  # type: ignore[misc]
    """Small modal that returns ``True`` only when the user approves."""

    CSS = """
    ConfirmationModal {
        align: center middle;
    }
    #confirm-dialog {
        width: 80;
        max-width: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #confirm-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #confirm-body {
        height: auto;
        margin-bottom: 1;
    }
    #confirm-actions {
        align-horizontal: right;
        height: auto;
    }
    Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, request: ConfirmationRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self):  # pragma: no cover - requires Textual runtime
        if Container is None or Static is None or Horizontal is None or Button is None:
            return
        with Container(id="confirm-dialog"):
            yield Static(self.request.title, id="confirm-title")
            yield Static(self.request.body, id="confirm-body")
            with Horizontal(id="confirm-actions"):
                yield Button(self.request.cancel_label, id="cancel", variant="default")
                yield Button(self.request.confirm_label, id="confirm", variant="primary")

    def on_button_pressed(self, event: Any) -> None:  # pragma: no cover - requires Textual runtime
        button_id = getattr(getattr(event, "button", None), "id", "")
        self.dismiss(button_id == "confirm")

    def action_cancel(self) -> None:  # pragma: no cover - requires Textual runtime
        self.dismiss(False)


async def confirm_with_app(
    app: Any,
    request: ConfirmationRequest,
    *,
    default: bool = False,
) -> bool:
    """Ask for confirmation through a Textual app when possible."""
    push_wait = getattr(app, "push_screen_wait", None)
    if not callable(push_wait) or Button is None:
        return default
    try:
        return bool(await push_wait(ConfirmationModal(request)))
    except Exception:
        return default


async def confirm(
    request: ConfirmationRequest,
    *,
    default: bool = False,
    app: Any = None,
) -> bool:
    """Return a conservative default unless an app-backed modal is available."""
    if app is not None:
        return await confirm_with_app(app, request, default=default)
    return default


def auto_merger_confirm_fn(app: Any):
    """Build an ``AutoMerger`` confirmation callback backed by the TUI."""

    async def _confirm(assessment: Any, patch: Any) -> bool:
        request = ConfirmationRequest(
            title=f"Auto-improvement: {getattr(patch, 'target_skill', '') or 'patch'}",
            body=_patch_review_body(assessment, patch),
            confirm_label="Approve patch",
            cancel_label="Reject",
        )
        return await confirm_with_app(app, request, default=False)

    return _confirm


def _patch_review_body(assessment: Any, patch: Any, *, max_chars: int = 2400) -> str:
    """Build a compact review body with risk, eval summary, and diff preview."""
    risk = getattr(getattr(assessment, "risk", ""), "value", getattr(assessment, "risk", ""))
    metadata = getattr(patch, "metadata", {}) or {}
    eval_summary = metadata.get("eval_summary") or metadata.get("eval") or ""
    diff = str(getattr(patch, "unified_diff", "") or "")
    if len(diff) > max_chars:
        diff = diff[:max_chars].rstrip() + "\n... <diff truncated for modal>"
    blocks = [
        f"Risk: {risk}",
        f"Reason: {getattr(assessment, 'reason', '')}",
        f"Target: {getattr(patch, 'target_path', '')}",
    ]
    if eval_summary:
        blocks.append(f"Eval: {eval_summary}")
    if diff:
        blocks.extend(["", "Diff preview:", diff])
    blocks.extend(["", "Approve only if the diff and eval summary match the intended repair."])
    return "\n".join(blocks)


__all__ = [
    "ConfirmationModal",
    "ConfirmationRequest",
    "auto_merger_confirm_fn",
    "confirm",
    "confirm_with_app",
    "_patch_review_body",
]
