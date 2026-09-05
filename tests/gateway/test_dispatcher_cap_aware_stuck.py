"""Tests for the cap-aware stuck-dispatcher warning gate.

Added 2026-08-23: the dispatcher previously fired "kanban dispatcher
stuck: ready queue non-empty" warnings even when the only skip reason
was per-profile-cap (sibling worker mid-flight). The warning is correct
in spirit (something ISN'T spawning despite ready tasks) but the
situation is intentional throttling, not a stuck dispatcher. The fix
tracks ``cap_blocked_all`` per tick: if every board's skip is purely
``skipped_per_profile_capped``, reset ``bad_ticks`` so the warning
doesn't accumulate.

These tests verify the source-level behavior without spinning up the
full gateway: we read the source and assert the gate logic is present.
End-to-end verification happens via the live gateway log — when no
sibling worker is mid-flight the warning fires normally; when a
sibling is mid-flight (e.g. a long-running spritefactory task) it
should stay silent.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _dispatch_result_is_expected_idle,
)


def _dispatcher_watcher_source() -> str:
    return inspect.getsource(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_cap_blocked_all_flag_exists():
    """The fix introduces a ``cap_blocked_all`` accumulator."""
    assert "cap_blocked_all" in _dispatcher_watcher_source()


def test_bad_ticks_gate_excludes_cap_blocked_all():
    """``bad_ticks`` must NOT increment when every board's skip is purely cap."""
    src = _dispatcher_watcher_source()
    # Find the conditional that gates the bad_ticks increment.
    pattern = re.compile(
        r"if\s+ready_pending\s+and\s+not\s+any_spawned.*?\bbad_ticks\s*\+=\s*1",
        re.DOTALL,
    )
    matches = pattern.findall(src)
    assert matches, "expected the bad_ticks-increment conditional to exist"
    # At least one match must include the cap_blocked_all guard.
    assert any(
        "cap_blocked_all" in m for m in matches
    ), "bad_ticks increment must be gated by ``not cap_blocked_all``"


def test_cap_blocked_all_initialized_per_tick():
    """``cap_blocked_all`` must be reset at the top of every tick (not module-level)."""
    src = _dispatcher_watcher_source()
    # Look for ``cap_blocked_all = True`` inside the tick body.
    assert "cap_blocked_all = True" in src
    # And it must be followed by the per-board loop (not in the wrong scope).
    pattern = re.compile(
        r"cap_blocked_all\s*=\s*True.*?for\s+\w+,\s*\w+\s+in\s*\(.+\):",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "cap_blocked_all=True must precede the per-board loop, not be at module scope"
    )


def test_cap_blocked_all_reset_on_spawn():
    """When any board actually spawns, cap_blocked_all must be False so a future
    board with a non-cap reason doesn't get masked."""
    src = _dispatcher_watcher_source()
    # Look for the path where ``any_spawned = True`` triggers
    # ``cap_blocked_all = False`` (a guard against a later board in the same
    # tick having a non-cap reason but being masked by an earlier cap).
    assert "any_spawned = True" in src
    # The cap_blocked_all = False line should be adjacent to any_spawned = True.
    pattern = re.compile(
        r"any_spawned\s*=\s*True\s*\n\s*cap_blocked_all\s*=\s*False",
    )
    assert pattern.search(src), (
        "any_spawned=True must immediately reset cap_blocked_all=False"
    )


def test_skipped_per_profile_capped_classified_correctly():
    """The cap-aware classification logic should look at the per-board
    DispatchResult's ``skipped_per_profile_capped`` field and treat ONLY
    cap-skips as 'correctly idle'."""
    src = inspect.getsource(_dispatch_result_is_expected_idle)
    assert "skipped_per_profile_capped" in src
    # The skip-reason check should NOT treat unassigned / nonspawnable as cap.
    # Find the variable that holds the per-board cap count.
    assert re.search(r"skipped_capped\s*=\s*list\s*\(", src), (
        "per-board cap count must be captured into ``skipped_capped``"
    )
    # The combined "other skips" count should include unassigned / nonspawnable
    # so those still fire the warning.
    assert "skipped_unassigned" in src
    assert "skipped_nonspawnable" in src


def test_no_regression_in_pause_short_circuit():
    """The pause-short-circuit (``bad_ticks = 0``) must still work."""
    src = _dispatcher_watcher_source()
    assert "ready_pending = False" in src
    assert "bad_ticks = 0" in src


def test_disk_governor_block_is_classified_as_expected_idle():
    """RED/UNKNOWN disk admission is intentional throttling, not stuck."""
    for pressure in ("RED", "UNKNOWN"):
        result = SimpleNamespace(
            disk_pressure=pressure,
            skipped_per_profile_capped=[],
            skipped_unassigned=[],
            skipped_nonspawnable=[],
            reclaimed=0,
            crashed=[],
            timed_out=[],
        )
        assert _dispatch_result_is_expected_idle(result) is True


def test_disk_governor_green_or_unset_does_not_mask_stuck_ready_work():
    """Only blocking governor states reset the stuck-warning accumulator."""
    for pressure in (None, "GREEN", "YELLOW"):
        result = SimpleNamespace(
            disk_pressure=pressure,
            skipped_per_profile_capped=[],
            skipped_unassigned=[],
            skipped_nonspawnable=[],
            reclaimed=0,
            crashed=[],
            timed_out=[],
        )
        assert _dispatch_result_is_expected_idle(result) is False


# ---------------------------------------------------------------------------
# Orphan-assignee / non-spawnable classification (t_67ec48d0, 2026-09-05)
# ---------------------------------------------------------------------------
#
# Before the fix: ``skipped_nonspawnable`` was counted in
# ``skipped_others``, so a board whose only skip reason was a queue of
# orphan / terminal-lane assignees (e.g. ``gpt-5.6-luna`` slipped past
# intake, or a steady-state ``orion-cc`` queue) tripped
# ``cap_blocked_all = False`` and the dispatcher cried wolf every 5 min
# for as long as the queue stayed full.  ``kanban_db.py`` already
# documented ``skipped_nonspawnable`` as "expected steady-state on
# multi-lane setups; NOT an operator-actionable failure" — the helper
# just wasn't honoring that docstring.
#
# The fix also corrected a latent crash: ``DispatchResult.reclaimed`` is
# declared ``int = 0`` (a count) but the helper did
# ``len(getattr(result, "reclaimed", []) or [])`` — when a stale-claim
# requeue happened on the same tick, ``reclaimed`` was a non-zero int
# and ``len(1)`` raised ``TypeError``. The watcher swallowed the
# exception, ``bad_ticks`` never reset, and the warning continued to
# fire on every port-cap tick until the gateway restarted.  The helper
# now reads ``reclaimed`` as an int.


def _result(**overrides):
    """Build a minimal ``DispatchResult``-like SimpleNamespace for tests."""
    base = dict(
        disk_pressure=None,
        skipped_per_profile_capped=[],
        skipped_unassigned=[],
        skipped_nonspawnable=[],
        reclaimed=0,
        crashed=[],
        timed_out=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_nonspawnable_only_is_expected_idle():
    """A board whose only skip is ``skipped_nonspawnable`` is correctly idle.

    Multi-lane setups keep the ready queue full of terminal-lane cards
    (``orion-cc`` / ``orion-research`` / ``orion-sdlc-review``) that the
    dispatcher correctly refuses to spawn — terminals pull those via
    ``claim_task`` directly. The stuck-warning has to stay silent in
    that state, otherwise it becomes a 5-min nag on every busy portfolio.
    """
    res = _result(skipped_nonspawnable=["t_a", "t_b"])
    assert _dispatch_result_is_expected_idle(res) is True


def test_cap_alone_still_expected_idle():
    """Regression: a pure per-profile-cap skip still returns True."""
    res = _result(skipped_per_profile_capped=[("t_a", "overwatch", 2)])
    assert _dispatch_result_is_expected_idle(res) is True


def test_cap_plus_nonspawnable_is_NOT_expected_idle():
    """Cap + nonspawnable together is NOT pure: defer to the warning so the
    operator can see mixed-skip drift if it becomes the steady state.
    """
    res = _result(
        skipped_per_profile_capped=[("t_a", "overwatch", 2)],
        skipped_nonspawnable=["t_b"],
    )
    assert _dispatch_result_is_expected_idle(res) is False


def test_nonspawnable_plus_crash_is_NOT_expected_idle():
    """A crash alongside nonspawnable work is real — keep the warning."""
    res = _result(
        skipped_nonspawnable=["t_a"],
        crashed=["t_b"],
    )
    assert _dispatch_result_is_expected_idle(res) is False


def test_reclaimed_int_does_not_crash():
    """Regression: ``reclaimed`` is ``int``, not a list.

    Pre-fix this raised ``TypeError: object of type 'int' has no len()``
    on every tick where a stale-claim requeue happened, which pinned
    ``bad_ticks`` and made the warning fire on every port-cap tick until
    the gateway restarted (verified against conductor's gateway.error.log
    on 2026-09-04 19:54:34Z traceback).
    """
    res = _result(reclaimed=1)  # int, not []
    assert _dispatch_result_is_expected_idle(res) is False  # not expected idle


def test_reclaimed_int_zero_is_expected_idle_when_only_cap():
    """``reclaimed=0`` with only a cap skip is still pure cap = expected idle."""
    res = _result(
        skipped_per_profile_capped=[("t_a", "overwatch", 2)],
        reclaimed=0,
    )
    assert _dispatch_result_is_expected_idle(res) is True


def test_reclaimed_int_nonzero_alongside_nonspawnable_is_NOT_idle():
    """When the queue has ONLY nonspawnable work but ``reclaimed > 0``
    this tick, a stale worker had to be reclaimed — that's a real signal
    even if the spawn path is correctly idle, so the warning keeps firing
    to surface the worker-staleness drift. (``reclaimed`` is the count
    of stale-claim requeues; non-zero means at least one worker died or
    went silent — operator-actionable regardless of ready-queue state.)
    """
    res = _result(
        skipped_nonspawnable=["t_a"],
        reclaimed=2,
    )
    assert _dispatch_result_is_expected_idle(res) is False


def test_reclaimed_int_with_real_stuck_signals_still_warns():
    """A real crash alongside any cap-or-nonspawnable still trips the warning."""
    res = _result(
        skipped_nonspawnable=["t_a"],
        reclaimed=1,
        crashed=["t_b"],
    )
    assert _dispatch_result_is_expected_idle(res) is False


def test_skipped_unassigned_alone_still_warns():
    """Regression: empty-assignee tasks remain operator-actionable.

    ``skipped_unassigned`` (no assignee at all) is a different bucket
    from ``skipped_nonspawnable`` (assignee exists but doesn't map to a
    real profile). Empty-assignee tasks usually mean a misfiled card
    that needs human routing — the warning should still fire so the
    operator notices.
    """
    res = _result(skipped_unassigned=["t_a"])
    assert _dispatch_result_is_expected_idle(res) is False
