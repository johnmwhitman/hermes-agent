"""Admission-control tests for repeated ticks on one shared Kanban board.

The gateway's machine-global ``.dispatcher.lock`` elects one dispatcher
owner.  That owner resolves its concurrency policy at boot and passes the
same explicit caps to every ``dispatch_once`` call.  ``dispatch_once`` is a
per-call primitive: it enforces the supplied caps against durable running
state, but deliberately does not persist caller configuration in the board
database (and a dry run deliberately writes no running state).
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_multi(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + platformops/conductor."""
    test_home = tempfile.mkdtemp(prefix="kanban_multi_dispatcher_test_")
    for prof in ("platformops", "conductor", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    """Spawn stub: pretend the worker started and got a PID."""
    return 99999


def test_repeated_owner_ticks_share_explicit_strict_policy(isolated_kanban_home_multi):
    """The elected owner supplies the same caps on every tick.

    The first tick may start one platformops worker.  The second tick observes
    that durable running row and must start none, preserving both the global
    and per-profile cap across ticks without any hidden policy persistence.
    """
    kb = isolated_kanban_home_multi
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Shared")
        # 5 platformops tasks queued, priority identical so order is FIFO.
        for i in range(5):
            kb.create_task(conn, title=f"po{i}", assignee="platformops")

    # Use this test process as the worker PID so the second tick sees a live
    # worker rather than reclaiming the fixture as crashed.
    def _live_fake_spawn(*args, **kwargs):
        return os.getpid()

    with kb.connect_closing() as conn:
        first = kb.dispatch_once(
            conn, spawn_fn=_live_fake_spawn,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )
        first_count = sum(1 for s in first.spawned if s[1] == "platformops")

    with kb.connect_closing() as conn:
        second = kb.dispatch_once(
            conn, spawn_fn=_live_fake_spawn,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )
        second_count = sum(1 for s in second.spawned if s[1] == "platformops")

    assert first_count == 1
    assert second_count == 0


def test_dry_run_cap_policy_is_per_call(isolated_kanban_home_multi):
    """A dry run neither claims tasks nor persists its caller's policy.

    This prevents one CLI preview from silently changing the elected gateway
    owner's later policy.  A capped preview sees one eligible task; a later
    uncapped preview still sees all three because neither call mutates state.
    """
    kb = isolated_kanban_home_multi
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Shared")
        for i in range(3):
            kb.create_task(conn, title=f"po{i}", assignee="platformops")

    with kb.connect_closing() as conn:
        res1 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=1,
        )
        first_count = sum(1 for s in res1.spawned if s[1] == "platformops")

    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
        )
        second_count = sum(1 for s in res2.spawned if s[1] == "platformops")

    assert first_count == 1
    assert second_count == 3


def test_global_cap_wins_when_per_profile_unset(isolated_kanban_home_multi):
    """The strict cap can be EITHER max_in_progress OR
    max_in_progress_per_profile; the smaller of the two wins for the
    affected assignee. With one dispatcher setting max_in_progress=1
    and per_profile=None, total running must be <=1 regardless of how
    many profiles are queued."""
    kb = isolated_kanban_home_multi
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Shared")
        for i in range(4):
            kb.create_task(conn, title=f"po{i}", assignee="platformops")
        for i in range(2):
            kb.create_task(conn, title=f"co{i}", assignee="conductor")

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress=1,  # strict global cap, no per-profile override
        )

    total_spawned = len(res.spawned)
    assert total_spawned <= 1, (
        f"global max_in_progress=1 was violated: dispatched {total_spawned}; "
        "expected <=1 because the elected owner supplied a strict global cap."
    )
