"""Regression tests for the release-or-record actor after
``complete_task`` → ``recompute_ready`` (conductor proposal on
t_add33d23, card t_e6423d36).

The gap: ``recompute_ready`` honours sticky blocks (#28712) by leaving
any task whose most recent ``blocked``/``unblocked`` event is
``blocked`` in the ``blocked`` column forever — even when the block
carried NO reason and every parent has since completed.  The board
measured 35 such silently-stranded cards on 2026-09-02; the conductor's
manual pass (t_c62f0927) regrew to 1 within two hours.

The actor: when a blocked task's parents are all ``done``/``archived``
and its latest ``blocked`` event has no non-empty ``reason``, the task
is released (``unblocked`` event with ``reason='parents done;
auto-released'``) instead of being skipped by the sticky guard.  A
block WITH a reason is an explicit human hold and stays sticky —
upstream #28712 behaviour is unchanged.

Pinned here:

* A child sticky-blocked with no reason whose last parent completes is
  auto-released to its resume status by ``recompute_ready`` (called by
  ``complete_task``).
* The release writes an ``unblocked`` event carrying the auto-release
  reason, so the card is never silently stranded.
* A child sticky-blocked WITH an explicit reason is untouched.
* A blocked child with no reason whose parents are NOT all done stays
  blocked (the actor only fires on parent-satisfaction).
* Circuit-breaker (non-sticky) blocks still auto-recover exactly as
  before.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _latest_block_unblock_event(conn, task_id: str) -> tuple[str, dict]:
    row = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert row is not None, "expected at least one blocked/unblocked event"
    payload = json.loads(row["payload"]) if row["payload"] else {}
    return row["kind"], payload


def _make_parent_child(conn, *, block_reason):
    """Create parent+child, run the child, and sticky-block it with the
    given reason (None / '' / whitespace → reasonless)."""
    parent = kb.create_task(conn, title="parent")
    child = kb.create_task(conn, title="child", parents=[parent])
    # Child cannot be claimed while the parent is undone; simulate the
    # worker having run and blocked by flipping the parent through its
    # lifecycle first.
    kb.claim_task(conn, parent)
    kb.complete_task(conn, parent, summary="parent done")
    # Now the child is ready (recompute_ready ran inside complete_task).
    assert kb.get_task(conn, child).status == "ready"
    kb.claim_task(conn, child)
    run_id = kb.get_task(conn, child).current_run_id
    assert kb.block_task(
        conn, child, reason=block_reason, expected_run_id=run_id
    )
    assert kb.get_task(conn, child).status == "blocked"
    return parent, child


# ---------------------------------------------------------------------------
# RED→GREEN: reasonless sticky block on a parent-satisfied child is released
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_reason", [None, "", "   "])
def test_reasonless_sticky_block_is_auto_released_when_parents_done(
    kanban_home: Path, empty_reason
) -> None:
    """The exact stranded shape from t_c62f0927: a blocked child whose
    latest blocked event has no usable reason and whose parents are all
    done must not sit in the board forever.  ``recompute_ready`` (as
    invoked by the completing parent's ``complete_task``) releases it
    and records why."""
    with kb.connect() as conn:
        parent, child = _make_parent_child(conn, block_reason=empty_reason)

        promoted = kb.recompute_ready(conn)
        assert promoted >= 1, "reasonless parent-satisfied block must release"
        task = kb.get_task(conn, child)
        assert task.status in ("ready", "review")

        kind, payload = _latest_block_unblock_event(conn, child)
        assert kind == "unblocked", (
            "the release must be recorded as an 'unblocked' event so the "
            "card is never silently stranded"
        )
        assert payload.get("reason") == "parents done; auto-released"


# ---------------------------------------------------------------------------
# GREEN guard: explicit hold reasons stay sticky (upstream #28712 intact)
# ---------------------------------------------------------------------------


def test_sticky_block_with_reason_is_untouched(kanban_home: Path) -> None:
    """A worker/operator block carrying a real reason is a deliberate
    human handoff — the release-or-record actor must not touch it."""
    with kb.connect() as conn:
        parent, child = _make_parent_child(
            conn, block_reason="needs_input: awaiting API key from John"
        )

        for _ in range(3):
            promoted = kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "blocked"

        kind, payload = _latest_block_unblock_event(conn, child)
        assert kind == "blocked"
        assert payload.get("reason") == "needs_input: awaiting API key from John"


# ---------------------------------------------------------------------------
# GREEN guard: parents not done → no release, even with no reason
# ---------------------------------------------------------------------------


def test_reasonless_block_stays_blocked_while_parents_undone(
    kanban_home: Path,
) -> None:
    """The actor fires only on parent-satisfaction.  A reasonless block
    with live parents is a legitimate wait and must not be flipped."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="slow parent")
        sibling = kb.create_task(conn, title="sibling still running")
        child = kb.create_task(
            conn, title="child", parents=[parent, sibling]
        )
        # Complete one parent; the sibling parent stays undone.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="one parent done")
        assert kb.get_task(conn, child).status == "todo"

        # Force the stranded shape the way the 35 measured cards got
        # there: a reasonless sticky ``blocked`` event plus a ``blocked``
        # status on a child that still has a live parent.
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'blocked', ?, ?)",
            (child, json.dumps({"reason": None, "kind": None}), now),
        )
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (child,)
        )
        conn.commit()
        assert kb.get_task(conn, child).status == "blocked"

        for _ in range(3):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "blocked", (
                "undone parents → the actor must not release"
            )


# ---------------------------------------------------------------------------
# GREEN guard: create_task(initial_status='blocked') human-ops park stays
# sticky even though its blocked event carries no reason
# ---------------------------------------------------------------------------


def test_initial_status_blocked_park_is_not_auto_released(
    kanban_home: Path,
) -> None:
    """``create_task(initial_status='blocked')`` records
    ``{"initial": True}`` with no reason — the park IS the reason.  The
    release-or-record actor must not touch it (sibling pin:
    test_kanban_disk_governor_guard.py)."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="human-ops park", initial_status="blocked"
        )
        for _ in range(3):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, tid).status == "blocked"
        kind, payload = _latest_block_unblock_event(conn, tid)
        assert kind == "blocked"
        assert payload.get("initial") is True


# ---------------------------------------------------------------------------
# GREEN guard: circuit-breaker (non-sticky) blocks still auto-recover
# ---------------------------------------------------------------------------


def test_circuit_breaker_block_still_auto_recovers(kanban_home: Path) -> None:
    """A blocked task with NO blocked/unblocked event at all (the
    circuit-breaker shape: status flipped by ``_record_task_failure``)
    keeps its pre-#28712 auto-recover semantics — the actor only
    narrows the sticky-guard skip, never the recovery path."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        # Circuit-breaker shape: blocked status, no 'blocked' event.
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (child,)
        )
        conn.commit()

        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="done")
        assert kb.get_task(conn, child).status == "ready"
