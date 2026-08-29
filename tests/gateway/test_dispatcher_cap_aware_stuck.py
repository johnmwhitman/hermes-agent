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
            reclaimed=[],
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
            reclaimed=[],
            crashed=[],
            timed_out=[],
        )
        assert _dispatch_result_is_expected_idle(result) is False
