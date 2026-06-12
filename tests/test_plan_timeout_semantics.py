"""Plan-timeout semantics: the SIGALRM-raised deadlines must propagate through the runtime's broad
``except Exception`` guards, so they inherit BaseException (like KeyboardInterrupt), not Exception."""

from __future__ import annotations

import pytest

from biobank_agent.cli.interactive import (
    PlanBuildTimeoutError,
    PlanStepTimeoutError,
    mark_activity,
)


@pytest.mark.parametrize("exc_type", [PlanStepTimeoutError, PlanBuildTimeoutError])
def test_plan_timeout_bypasses_broad_except(exc_type):
    # Inherits BaseException, NOT Exception — so the real runtime's tool/agent-loop `except Exception`
    # guards cannot swallow the deadline before the explicit `except PlanStepTimeoutError` handler runs.
    assert issubclass(exc_type, BaseException)
    assert not issubclass(exc_type, Exception)

    swallowed_by_broad = False
    with pytest.raises(exc_type):
        try:
            raise exc_type("deadline")
        except Exception:  # noqa: BLE001 - intentionally broad, mirrors the runtime guards
            swallowed_by_broad = True
    assert not swallowed_by_broad, f"{exc_type.__name__} was swallowed by except Exception"


def test_mark_activity_is_safe_noop_when_idle():
    # No active watchdog -> heartbeat is a harmless no-op (must never raise).
    mark_activity()
