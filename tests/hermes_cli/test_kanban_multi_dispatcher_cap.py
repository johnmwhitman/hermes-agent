"""Regression test for t_bd64ac2d — multi-dispatcher admission control.

Root cause (2026-08-28): ``kanban.max_in_progress`` and
``kanban.max_in_progress_per_profile`` are enforced ONLY when the dispatcher
that runs the tick passes them in. Every gateway that boots with
``dispatch_in_gateway=true`` (the upstream default) races the singleton
``.dispatcher.lock``; the FIRST one to win becomes the de-facto admission
authority for the whole shared board. If that gateway's profile has no
``kanban.max_in_progress*`` keys (e.g. researcher's config), caps become
None and ``dispatch_once`` happily spawns past every operator-configured
limit. The board-scoped dispatch lock (``_dispatch_tick_lock``) only
serialises writers — it does not unify policy.

This fixture proves that ``dispatch_once`` itself (the lowest-level
admission-control primitive) correctly enforces an EITHER-side cap when the
caller passes them in. The DISPATCHER-OWNER contract test that proves the
singleton wins is in a sibling test
(``test_kanban_dispatcher_owner_authority.py``); the fix must install that
test once the durable repair lands.

This is the deterministic RED fixture the operator requested. It fails on
unpatched code: two callers invoking ``dispatch_once`` back-to-back with
ONE cap set and the other absent let the absent cap dispatch past the
configured strict cap, exactly the live reproduction.
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


def test_two_dispatchers_different_caps_share_strict_policy(isolated_kanban_home_multi):
    """Two callers invoke ``dispatch_once`` against the same shared DB with
    different cap configurations. The STRICT-shared effective cap must be
    honoured regardless of which dispatcher is currently holding the
    singleton.

    Reproduction of t_bd64ac2d:
      - caller A (researcher): max_in_progress_per_profile=None
        (no caps configured) → no per-profile enforcement
      - caller B (conductor): max_in_progress=2, max_in_progress_per_profile=1
        (operator intent)

    Without the fix, each caller's tick enforces ONLY its own caps. With
    the fix, the strict-shared policy (B's caps) wins, so the platformops
    ready backlog is bounded to N=1 even when caller A is the singleton.
    """
    kb = isolated_kanban_home_multi
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Shared")
        # 5 platformops tasks queued, priority identical so order is FIFO.
        for i in range(5):
            kb.create_task(conn, title=f"po{i}", assignee="platformops")

    # Tick 1: caller A (researcher) holds the singleton; no caps.
    # Tick 2: caller B (conductor) runs with strict caps.
    with kb.connect_closing() as conn:
        a_res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            # dispatcher A's config: no caps
        )
        a_count = sum(1 for s in a_res.spawned if s[1] == "platformops")

    with kb.connect_closing() as conn:
        b_res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )
        b_count = sum(1 for s in b_res.spawned if s[1] == "platformops")

    # STRICT-shared policy (B's caps) is the operator-configured truth.
    # Caller A's lack of caps must NOT relax it.
    assert a_count <= 1, (
        f"caller A (no caps) dispatched {a_count} platformops workers; "
        "expected <=1 because the strict-shared per-profile cap is 1. "
        "This is the t_bd64ac2d regression."
    )
    assert b_count <= 1, (
        f"caller B (cap=1) dispatched {b_count} platformops workers; "
        "expected <=1 per per-profile cap."
    )


def test_strict_cap_wins_when_one_dispatcher_omits_caps(isolated_kanban_home_multi):
    """Three ready tasks for platformops; one dispatcher with cap=1
    followed by a dispatcher with no caps. The strict cap (1) must hold
    across both ticks because the caps are operator-configured truth,
    not per-tick hints."""
    kb = isolated_kanban_home_multi
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Shared")
        for i in range(3):
            kb.create_task(conn, title=f"po{i}", assignee="platformops")

    # Tick 1: caller with caps (cap=1)
    with kb.connect_closing() as conn:
        res1 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=1,
        )
        first_count = sum(1 for s in res1.spawned if s[1] == "platformops")

    # Tick 2: caller with no caps (simulates researcher owning the singleton)
    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            # no caps passed
        )
        second_count = sum(1 for s in res2.spawned if s[1] == "platformops")

    assert first_count == 1, (
        f"tick 1 (cap=1) dispatched {first_count}; expected exactly 1"
    )
    # The fix MUST persist the strict cap so tick 2 (caller without caps)
    # also respects it. Without the fix, tick 2 sees no running-count
    # bookkeeping (it doesn't track per-profile cap) and spawns the rest.
    assert second_count == 0, (
        f"tick 2 (no caps) dispatched {second_count}; expected 0 because "
        "tick 1 already claimed 1 and strict per-profile cap is 1. "
        "This proves the missing dispatcher-pinning of strict policy."
    )


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
        "expected <=1. Without the fix this test still passes (current code "
        "honours global cap), but the per-profile + global interaction is "
        "what t_bd64ac2d exposes."
    )