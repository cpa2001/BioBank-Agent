"""Tests for the interactive menu input layer.

These pin the root cause of the "arrow keys repeat the display" bug: the old
``_read_single_key`` used buffered ``sys.stdin.read(1)`` + ``select`` on the fd,
so a 3-byte arrow burst (``\\x1b[A``) was split into three non-matching keys.
The fix reads raw bytes with ``os.read`` and parses CSI/SS3 sequences precisely.
"""

from __future__ import annotations

import os
import pty
import re
import sys
import threading
import time
import tty
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from biobank_agent.cli.interactive import InteractiveShell, _CUSTOM_ANSWER_LABEL
from biobank_agent.progress import PlanRunDashboard

from tests.test_interactive_cli_runtime import (
    _FakeLegacyAgent,
    _FakeOrchestrator,
    _FakeProvider,
    _FakeTool,
    _Settings,
)


class _FakeStdin:
    """Minimal stand-in for ``sys.stdin`` backed by a real (pty) fd."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def isatty(self) -> bool:
        return True


def _shell(tmp_path: Path) -> InteractiveShell:
    shell = InteractiveShell(
        settings=_Settings(tmp_path),
        console=Console(file=StringIO(), force_terminal=False, width=120),
    )
    shell.legacy_agent_factory = _FakeLegacyAgent
    shell.initialize()
    shell.runtime.provider_router.providers = {"fake-model": _FakeProvider()}
    shell.runtime.tool_registry.register(_FakeTool())
    shell.legacy_agent.orchestrator = _FakeOrchestrator()
    return shell


def _attach_pty(monkeypatch, shell: InteractiveShell) -> int:
    """Wire the shell's key reader to a pty; return the master fd for writing."""
    master, slave = pty.openpty()
    tty.setraw(slave)  # no echo / no CR-NL translation, deterministic bytes
    monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))
    monkeypatch.setattr(shell, "_interactive_terminal", lambda: True)
    return master


def test_read_single_key_decodes_arrow_in_one_read(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    os.write(master, b"\x1b[A")
    assert shell._read_single_key() == "\x1b[A"


def test_read_single_key_does_not_slurp_following_enter(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    os.write(master, b"\x1b[B\r")  # down + enter, written as one burst
    assert shell._read_single_key() == "\x1b[B"
    assert shell._read_single_key() == "\r"


def test_read_single_key_plain_char(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    os.write(master, b"k")
    assert shell._read_single_key() == "k"


def test_read_single_key_ctrl_c_raises(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    os.write(master, b"\x03")
    with pytest.raises(KeyboardInterrupt):
        shell._read_single_key()


def test_read_single_key_non_tty_raises_without_reading(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    monkeypatch.setattr(shell, "_interactive_terminal", lambda: False)
    with pytest.raises(KeyboardInterrupt):
        shell._read_single_key()


def test_keyboard_select_down_then_enter_returns_second(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    choices = shell._plan_review_choices()
    os.write(master, b"\x1b[B\r")  # move to 2nd item, confirm
    chosen = shell._keyboard_select_plan_review_choice(choices)
    assert chosen.action == choices[1].action


def test_keyboard_select_enter_confirms_highlighted_not_checked(tmp_path, monkeypatch):
    """Regression: old code confirmed a separate ``checked`` index, so a bare
    Enter after moving picked the wrong row. Enter must confirm the highlight."""
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    choices = shell._plan_review_choices()
    os.write(master, b"\x1b[B\x1b[B\r")  # down, down, enter -> 3rd item
    chosen = shell._keyboard_select_plan_review_choice(choices)
    assert chosen.action == choices[2].action


def test_keyboard_select_digit_shortcut(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    choices = shell._plan_review_choices()
    os.write(master, b"2")  # jump straight to item 2
    chosen = shell._keyboard_select_plan_review_choice(choices)
    assert chosen.action == choices[1].action


def test_read_single_key_drains_bracketed_paste(tmp_path, monkeypatch):
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    os.write(master, b"\x1b[200~2\n\x1b[201~")  # a paste containing digit + enter
    assert shell._read_single_key() == ""  # swallowed, returns the ignored sentinel


def test_keyboard_select_ignores_pasted_payload(tmp_path, monkeypatch):
    """Pasted "2\\n" must NOT pick item 2 / confirm; only the real keys after the
    paste end-marker drive the selection."""
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    choices = shell._plan_review_choices()
    os.write(master, b"\x1b[200~2\n\x1b[201~\x1b[B\r")  # paste, then Down + Enter
    chosen = shell._keyboard_select_plan_review_choice(choices)
    assert chosen.action == choices[1].action


# ---------------------------------------------------------------------------
# Phase A regression tests for the LIVE-path bug (the coverage gap that hid it):
# the previous fix tested the selector in isolation, never together with the
# planning dashboard and never on a *terminal* console (force_terminal=True),
# so the broken in-place redraw on CJK menus + the dashboard/selector Live
# interaction were never exercised.
# ---------------------------------------------------------------------------

_CJK_QUESTION = {
    "header": "分组定义",
    "question": "两个表型组的具体构成是什么？",
    "options": [
        {"label": "病例 vs 健康对照", "description": "标准病例对照关联分析"},
        {"label": "不同白癜风亚型", "description": "如节段型 vs 非节段型，需亚型间比较"},
        {"label": "不同严重程度/分期", "description": "按定量或等级分组，可能用QT/ordinal模型"},
    ],
}


def _term_shell(tmp_path: Path, monkeypatch):
    """Shell whose console is a *terminal* (so Rich emits real cursor-control
    codes we can inspect) with key input wired to a pty. Returns (shell, buf,
    master_fd)."""
    shell = _shell(tmp_path)
    buf = StringIO()
    shell.console = Console(file=buf, force_terminal=True, width=100)
    master = _attach_pty(monkeypatch, shell)
    return shell, buf, master


def test_clarifier_menu_redraws_in_place_and_returns_highlight(tmp_path, monkeypatch):
    """On a terminal, Down+Enter must return the 2nd CJK option AND the menu must
    redraw *in place* (cursor-up control codes emitted) rather than re-printing /
    stacking. Wide CJK glyphs previously made Rich undercount lines so the stale
    highlight stayed on screen ("arrows don't move the selection")."""
    shell, buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\x1b[B\r")  # Down, Enter -> 2nd option
    answer = shell._plan_clarifier(_CJK_QUESTION)
    text = buf.getvalue()
    assert answer == "不同白癜风亚型"
    assert len(re.findall(r"\x1b\[\d*A", text)) > 0, "expected cursor-up (in-place redraw)"
    assert "不同白癜风亚型" in text and "Enter confirm" in text


def test_clarifier_menu_blocks_until_a_key_arrives(tmp_path, monkeypatch):
    """The clarification prompt must BLOCK for input. Symptom #3 was prompts
    flying past without blocking; verify the selector waits for a real key."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    out: dict[str, object] = {}

    def worker():
        try:
            out["answer"] = shell._plan_clarifier(_CJK_QUESTION)
        except BaseException as exc:  # noqa: BLE001 - record for assertion
            out["err"] = repr(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.4)
    assert t.is_alive(), "selector returned without waiting for a keypress"
    os.write(master, b"\r")  # confirm default (1st)
    t.join(2.0)
    assert not t.is_alive()
    assert out.get("answer") == "病例 vs 健康对照"


def test_clarifier_ctrl_c_propagates_to_cancel_plan(tmp_path, monkeypatch):
    """Ctrl-C during a clarification must propagate (cancel the plan), NOT be
    swallowed as a silent 'skip' (which produced the '0 resolved' blast-through
    when the user hit Ctrl-C to escape an unresponsive menu)."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\x03")  # Ctrl-C
    with pytest.raises(KeyboardInterrupt):
        shell._plan_clarifier(_CJK_QUESTION)


def test_clarifier_window_keeps_selection_visible_on_short_terminal(tmp_path, monkeypatch):
    """On a short terminal with more options than fit, the selected row must stay
    visible: the menu shows a sliding window CENTERED on the selection. A plain
    top-slice (the first clamp implementation) would push the highlighted row
    off-screen — so option-4's label never renders. Navigate to option 4 on an
    8-row terminal with 6 two-line options and assert its label appears."""
    shell = _shell(tmp_path)
    buf = StringIO()
    shell.console = Console(file=buf, force_terminal=True, width=80, height=8)
    master = _attach_pty(monkeypatch, shell)
    options = [{"label": f"选项{i}", "description": f"描述 {i} 的详细说明文本"} for i in range(6)]
    question = {"header": "many", "question": "选择一个分组？", "options": options}
    os.write(master, b"\x1b[B\x1b[B\x1b[B\x1b[B\r")  # Down x4 -> option 4, Enter
    answer = shell._plan_clarifier(question)
    text = buf.getvalue()
    assert answer == "选项4"
    assert "选项4" in text, "selected row was windowed off-screen (top-slice bug)"
    assert "Enter confirm" in text


def test_clarifier_window_keeps_last_option_visible(tmp_path, monkeypatch):
    """Boundary case: selecting the LAST option (window cannot expand downward)
    must still keep that row on screen on a short terminal."""
    shell = _shell(tmp_path)
    buf = StringIO()
    shell.console = Console(file=buf, force_terminal=True, width=80, height=8)
    master = _attach_pty(monkeypatch, shell)
    options = [{"label": f"选项{i}", "description": f"描述 {i} 的详细说明文本"} for i in range(6)]
    question = {"header": "many", "question": "选择一个分组？", "options": options}
    # Down x5 -> option 5 (the last *real* option; a synthetic "Type something"
    # row is appended after it now), Enter.
    os.write(master, b"\x1b[B\x1b[B\x1b[B\x1b[B\x1b[B\r")
    answer = shell._plan_clarifier(question)
    text = buf.getvalue()
    assert answer == "选项5"
    assert "选项5" in text, "last option windowed off-screen"
    assert "Enter confirm" in text


# --- Part 2: INLINE free-text editing inside the menu panel ------------------


def test_read_single_key_returns_tab(tmp_path, monkeypatch):
    """A plain Tab must surface as '\\t' so the menu's edit branch can fire."""
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)
    os.write(master, b"\t")
    assert shell._read_single_key() == "\t"


def test_clarifier_tab_inline_edit_prefills_and_returns_typed_text(tmp_path, monkeypatch):
    """Tab on a focused option opens an INLINE editor prefilled with that row's
    label; typed ASCII+CJK is appended and Enter returns the edited text — all
    inside the panel, with NO break-out 'refine answer' prompt."""
    shell, buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\t" + "abc病".encode("utf-8") + b"\r")
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "病例 vs 健康对照abc病"
    text = buf.getvalue()
    assert "refine answer" not in text          # no break-out prompt
    assert "Enter submit" in text               # the in-panel edit footer
    assert re.findall(r"\x1b\[\d*A", text)       # redrew in place (cursor-up codes)


def test_clarifier_custom_row_enter_enters_inline_edit(tmp_path, monkeypatch):
    """Enter on the synthetic custom row opens an empty INLINE editor; the typed
    text is forwarded."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    # 3 real options -> custom row is index 3: Down x3, Enter (open edit), type, Enter.
    os.write(master, b"\x1b[B\x1b[B\x1b[B\r" + "节段型".encode("utf-8") + b"\r")
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "节段型"


def test_clarifier_inline_backspace_deletes(tmp_path, monkeypatch):
    """Backspace (both \\x7f and \\x08) deletes the char before the cursor."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    # custom row -> empty edit -> type 'xyz' -> 2 backspaces -> 'x' -> Enter
    os.write(master, b"\x1b[B\x1b[B\x1b[B\r" + b"xyz" + b"\x7f\x08" + b"\r")
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "x"


def test_clarifier_inline_esc_cancels_back_to_menu(tmp_path, monkeypatch):
    """Esc cancels the inline edit (no answer) and returns to the menu; a
    following Enter confirms the focused option."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\t" + b"xyz" + b"\x1b" + b"\r")  # Tab, type, Esc, Enter
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "病例 vs 健康对照"


def test_clarifier_inline_empty_submit_returns_to_menu(tmp_path, monkeypatch):
    """Submitting an empty buffer must NOT answer; it exits edit back to the menu
    so a following Enter confirms the focused option."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    # Tab (prefill row-0 label, 10 chars) -> backspace it all -> Enter (empty,
    # exits edit) -> Enter (confirm row 0).
    os.write(master, b"\t" + b"\x7f" * 14 + b"\r\r")
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "病例 vs 健康对照"


def test_clarifier_inline_digits_inserted_not_jumped(tmp_path, monkeypatch):
    """Digits typed WHILE editing are inserted into the buffer, never treated as
    a menu jump."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\x1b[B\x1b[B\x1b[B\r" + b"123" + b"\r")  # custom row -> '123'
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "123"


def test_clarifier_inline_left_right_insert_midstring(tmp_path, monkeypatch):
    """←/→ move the cursor so text can be inserted mid-string."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    # custom row -> type 'ac' -> Left -> 'b' (between a and c) -> Enter
    os.write(master, b"\x1b[B\x1b[B\x1b[B\r" + b"ac" + b"\x1b[D" + b"b" + b"\r")
    answer = shell._plan_clarifier(_CJK_QUESTION)
    assert answer == "abc"


def test_clarifier_many_tab_esc_cancels_stay_stable(tmp_path, monkeypatch):
    """A long burst of Tab+Esc (open edit / cancel) must remain a single in-panel
    loop (no recursion, no stack growth, no break-out) and still answer cleanly."""
    import inspect
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\t\x1b" * 250 + b"\r")   # 250 open+cancel, then confirm row 0
    base = len(inspect.stack())
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(base + 120)
    try:
        answer = shell._plan_clarifier(_CJK_QUESTION)
    finally:
        sys.setrecursionlimit(old)
    assert answer == "病例 vs 健康对照"


def test_clarifier_inline_edit_escapes_markup(tmp_path, monkeypatch):
    """Typing '[' / '[/]' in the inline editor must not crash the render."""
    shell, _buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\x1b[B\x1b[B\x1b[B\r" + b"a[/]b[bold]" + b"\r")
    answer = shell._plan_clarifier(_CJK_QUESTION)  # must not raise MarkupError
    assert answer == "a[/]b[bold]"


def test_clarifier_footer_advertises_tab_and_custom_row(tmp_path, monkeypatch):
    shell, buf, master = _term_shell(tmp_path, monkeypatch)
    os.write(master, b"\r")  # confirm row 0 immediately
    shell._plan_clarifier(_CJK_QUESTION)
    text = buf.getvalue()
    assert "Tab edit" in text
    assert "以上都不是" in text  # the synthetic custom-answer row is shown
    assert "✎" not in text      # emoji removed


class _FakePlan:
    objective = "x"
    steps: list = []


def test_dashboard_lazy_starts_only_after_clarification(tmp_path, monkeypatch):
    """Decisive integration test: the planning dashboard must NOT be started
    while the clarification selector runs (the two-Live interaction the prior
    fix never exercised). Drive the real _draft_plan_with_live_progress with a
    fake update_plan that emits pre-planning phases, calls the clarifier, then
    emits a planning phase; assert the answer is captured and dashboard.start
    fires only AFTER clarification."""
    shell = _shell(tmp_path)
    master = _attach_pty(monkeypatch, shell)

    order: list[str] = []
    real_start = PlanRunDashboard.start

    def spy_start(self, *a, **k):
        order.append("dashboard-start")
        return real_start(self, *a, **k)

    monkeypatch.setattr(PlanRunDashboard, "start", spy_start)

    captured: dict[str, object] = {}

    def fake_update_plan(session, objective, *, emit, clarifier, **kw):
        emit("Preflight", status="success", message="114 tools available")
        emit("Clarification", status="running", message="checking for ambiguities")
        order.append("clarify-call")
        captured["answer"] = clarifier(_CJK_QUESTION)
        order.append("clarify-done")
        emit("Clarification", status="success", message="1 clarification(s) resolved")
        emit("Planning", status="running", message="drafting 3 candidate plans")
        return _FakePlan()

    monkeypatch.setattr(shell.runtime, "update_plan", fake_update_plan)
    for name in (
        "_mark_plan_review_ready",
        "_record_plan_action_graph",
        "_gather_plan_context",
        "_record_plan_review_graph",
        "_render_planning_progress",
        "_render_plan",
        "_render_plan_flow",
    ):
        monkeypatch.setattr(shell, name, lambda *a, **k: None)
    monkeypatch.setattr(shell.runtime, "save_session", lambda *a, **k: None)

    os.write(master, b"\x1b[B\r")  # Down, Enter -> 2nd option
    shell._draft_plan_with_live_progress("白癜风 WGS 表型差异分析")

    assert captured["answer"] == "不同白癜风亚型"
    assert "dashboard-start" in order, order
    assert order.index("clarify-done") < order.index("dashboard-start"), order
    assert order.index("clarify-call") < order.index("dashboard-start"), order


class _FakeStdout:
    """Stand-in for a real-TTY ``sys.stdout``: ``isatty()`` is True, writes are
    swallowed. Used to exercise the REAL ``_interactive_terminal()`` (which ANDs
    ``stdin.isatty()`` and ``stdout.isatty()``)."""

    def __init__(self) -> None:
        self.buffer = []

    def isatty(self) -> bool:
        return True

    def write(self, s):  # pragma: no cover - exercised only if Live redirects
        self.buffer.append(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover
        pass


def test_clarifier_menu_does_not_redirect_stdout_so_it_never_self_aborts(tmp_path, monkeypatch):
    """Regression for the live ``/plan`` crash.

    The clarification menu opens its OWN Rich ``Live``. Rich defaults
    ``redirect_stdout=True``, swapping ``sys.stdout`` for a ``FileProxy`` whose
    ``isatty()`` returns False even on a real TTY. ``_read_single_key`` gates on
    ``_interactive_terminal()`` (``stdin.isatty() and stdout.isatty()``); under
    the redirect that flips to False and the menu raises ``KeyboardInterrupt`` on
    itself before reading a single key — which then propagated uncaught to
    ``main()`` and crashed the CLI.

    This test drives ``_live_single_select`` with the REAL ``_interactive_terminal``
    (NOT monkeypatched to True, unlike ``_attach_pty``) over a tty stdin+stdout,
    and asserts the menu reads the key and returns WITHOUT aborting, and that
    ``sys.stdout`` is never a Rich ``FileProxy`` while keys are read. It FAILS on
    the pre-fix code (default redirect) and PASSES once the menu's ``Live`` sets
    ``redirect_stdout=False``.
    """
    from rich.file_proxy import FileProxy

    shell = _shell(tmp_path)
    # The menu's Live only redirects sys.stdout when its console IS a terminal
    # (Rich's Live._enable_redirect_io gates on console.is_terminal) — exactly the
    # real `biobank` case. The default test console is a non-terminal StringIO, so
    # force a terminal console here or the bug cannot reproduce.
    monkeypatch.setattr(shell, "console", Console(file=StringIO(), force_terminal=True, width=120))
    master, slave = pty.openpty()
    tty.setraw(slave)
    monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))
    monkeypatch.setattr(sys, "stdout", _FakeStdout())
    # Deliberately DO NOT patch shell._interactive_terminal — the real guard must
    # observe the (un)redirected stdout for this regression to mean anything.

    seen_stdout_types: list[str] = []
    real_read = shell._read_single_key

    def spy_read():
        seen_stdout_types.append(type(sys.stdout).__name__)
        return real_read()

    monkeypatch.setattr(shell, "_read_single_key", spy_read)

    os.write(master, b"\r")  # confirm the highlighted (first) item

    try:
        idx = shell._live_single_select(
            title="t",
            header="pick one",
            items=[{"label": "a"}, {"label": "b"}],
            render_item=lambda item, is_sel, i: str(item["label"]),
        )
    except KeyboardInterrupt:
        pytest.fail(
            "menu self-aborted: the Live redirected sys.stdout to a FileProxy, "
            "flipping _interactive_terminal() to False (the /plan crash)."
        )

    assert idx == 0
    assert seen_stdout_types, "the read loop never ran"
    assert all(t != "FileProxy" for t in seen_stdout_types), seen_stdout_types
    assert not isinstance(sys.stdout, FileProxy)


def test_run_loop_continues_after_keyboardinterrupt_in_handle_line(tmp_path, monkeypatch):
    """A KeyboardInterrupt raised inside command handling (e.g. a genuine Ctrl-C
    during ``/plan``) must cancel the command and return to the prompt — never
    crash the REPL — while EOF still exits cleanly."""
    shell = _shell(tmp_path)
    monkeypatch.setattr(shell, "initialize", lambda **k: None)
    monkeypatch.setattr(shell, "startup_banner", lambda: "")

    lines = iter(["/plan something"])

    def fake_read_line():
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    calls = {"n": 0}

    def fake_handle_line(line):
        calls["n"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(shell, "_read_line", fake_read_line)
    monkeypatch.setattr(shell, "handle_line", fake_handle_line)

    shell.run()  # must return normally, not propagate KeyboardInterrupt

    out = shell.console.file.getvalue()
    assert calls["n"] == 1
    assert "Cancelled" in out, out
    assert "Goodbye" in out, out
