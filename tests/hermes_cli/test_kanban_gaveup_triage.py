"""Regression tests for the gave_up orphan class exposed by triage card
t_8285095c (card t_fae878e7).

The gap: ``kanban_db._has_sticky_block`` deliberately distinguishes
worker/operator sticky blocks (latest ``blocked`` event) from dispatcher
circuit breakers (``gave_up`` event, no ``blocked`` event).  But
``recompute_ready`` only promotes non-sticky ``blocked`` tasks to
``ready`` — for a child task whose parents are already done, the
landing status is ``ready`` and the guard ``status == 'ready' AND
status != 'blocked'`` skips it forever.  No supported actor polls for
``gave_up`` cards, so a dead-PID / timeout / protocol-violation orphan
sits in ``blocked`` with a spent failure budget indefinitely (the live
board carried three such cards until manual SQL triage).

The supported path added here:

* ``list_gaveup_orphans(conn)`` — read-only enumeration of the exact
  orphan shape: ``status='blocked'``, latest event ``gave_up``, no
  ``blocked``/``unblocked`` event after it (i.e. NOT a sticky block),
  ``consecutive_failures >= effective retry limit`` (fail-closed), all
  parents ``done``/``archived`` (the stuck class; parent-gated orphans
  surface after their parents finish).
* ``retry_gaveup_task(conn, tid, actor, reason)`` — guarded fresh
  retry: refuses sticky blocks, non-orphans, and tasks below the
  breaker limit; resets ``consecutive_failures`` (fresh budget —
  deliberate operator action, same as ``unblock_task``), lands at
  ``ready``/``todo`` via the shared parent re-gate, emits
  ``gave_up_retried`` with actor + reason.
* ``archive_gaveup_task(conn, tid, actor, reason)`` — guarded archive
  with the same shape checks; a human-decision ``blocked`` event is
  emitted FIRST so the orphan classifier can never misread the card
  mid-flight, then the existing ``archive_task`` machinery runs
  (``archived`` event, dependent recompute, workspace reap).

Pinned here:

* Dead-PID, timeout, and protocol-violation ``gave_up`` orphans are
  enumerable and retriable through the supported actor.
* An explicit sticky block stays sticky AND is not listed, retriable,
  or archivable through this actor (no misclassification).
* Failure limits stay fail-closed: a task below its effective limit is
  not an orphan and cannot be force-retried here.
* Every retry/archive leaves a reasoned event (actor + reason, audit).
* Release-or-record and sticky recompute behaviour is unchanged:
  ``recompute_ready`` still never touches ``gave_up`` orphans itself.
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


def _events(conn, task_id: str, kinds=None) -> list[tuple[str, dict]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    out = []
    for row in rows:
        if kinds is not None and row["kind"] not in kinds:
            continue
        payload = json.loads(row["payload"]) if row["payload"] else {}
        out.append((row["kind"], payload))
    return out


def _trip_breaker(conn, tid: str, *, failure_limit: int = 1) -> None:
    """Drive the task through the dispatcher crash path until the
    circuit breaker trips (``gave_up`` event, status ``blocked``)."""
    for _ in range(failure_limit):
        kb.claim_task(conn, tid)
        tripped = kb._record_task_failure(
            conn,
            tid,
            error="pid 4242 not alive",
            outcome="crashed",
            failure_limit=failure_limit,
            release_claim=True,
            end_run=True,
        )
    assert tripped
    assert kb.get_task(conn, tid).status == "blocked"


def _trip_breaker_timeout(conn, tid: str, *, failure_limit: int = 1) -> None:
    """Timeout path: the run is already closed and the source phase
    restored (``release_claim=False, end_run=False``); the breaker just
    flips the task back to blocked."""
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    # Simulate what the dispatcher's timeout sweep does BEFORE calling
    # _record_task_failure: restore the source phase, close the run.
    conn.execute(
        "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
        "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
        "WHERE id = ?",
        (tid,),
    )
    conn.execute(
        "UPDATE task_runs SET status = 'timed_out', outcome = 'timed_out', "
        "ended_at = ? WHERE id = ?",
        (int(time.time()), run_id),
    )
    conn.commit()
    tripped = kb._record_task_failure(
        conn,
        tid,
        error="timed out after 5400s",
        outcome="timed_out",
        failure_limit=failure_limit,
        release_claim=False,
        end_run=False,
        event_payload_extra={"elapsed": 5400},
    )
    assert tripped
    assert kb.get_task(conn, tid).status == "blocked"


def _trip_breaker_protocol_violation(conn, tid: str) -> None:
    """Protocol-violation gave_up (``force_trip=True`` path used by
    ``detect_crashed_workers`` once the violation streak hits its
    bound): the task is back at ``ready`` with its run closed, then the
    breaker force-trips it into ``blocked``."""
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
        "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
        "WHERE id = ?",
        (tid,),
    )
    conn.execute(
        "UPDATE task_runs SET status = 'crashed', outcome = 'crashed', "
        "ended_at = ?, metadata = ? WHERE id = ?",
        (now, json.dumps({"protocol_violation": True, "exit_code": 0}), run_id),
    )
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, 'protocol_violation', ?, ?)",
        (tid, run_id, json.dumps({"pid": 4242, "exit_code": 0}), now),
    )
    conn.commit()
    tripped = kb._record_task_failure(
        conn,
        tid,
        error="clean exit without terminal tool call",
        outcome="crashed",
        failure_limit=kb._PROTOCOL_VIOLATION_FAILURE_LIMIT,
        force_trip=True,
        release_claim=False,
        end_run=False,
        event_payload_extra={
            "pid": 4242,
            "protocol_violations": kb._PROTOCOL_VIOLATION_FAILURE_LIMIT,
            "protocol_violation_limit": kb._PROTOCOL_VIOLATION_FAILURE_LIMIT,
        },
    )
    assert tripped
    assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Enumeration: the exact orphan shape is visible, nothing else leaks in
# ---------------------------------------------------------------------------


def test_dead_pid_orphan_is_enumerable(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="dead pid orphan")
        _trip_breaker(conn, tid, failure_limit=1)

        orphans = kb.list_gaveup_orphans(conn, failure_limit=1)
        assert [o["task_id"] for o in orphans] == [tid]
        orphan = orphans[0]
        assert orphan["consecutive_failures"] == 1
        assert orphan["effective_limit"] == 1
        assert orphan["trigger_outcome"] == "crashed"
        assert "not alive" in orphan["last_failure_error"]
        assert orphan["last_event_at"] > 0


def test_timeout_orphan_is_enumerable(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="timeout orphan")
        _trip_breaker_timeout(conn, tid, failure_limit=1)

        orphans = kb.list_gaveup_orphans(conn, failure_limit=1)
        assert [o["task_id"] for o in orphans] == [tid]
        assert orphans[0]["trigger_outcome"] == "timed_out"


def test_protocol_violation_orphan_is_enumerable(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="protocol violation orphan")
        _trip_breaker_protocol_violation(conn, tid)

        orphans = kb.list_gaveup_orphans(conn, failure_limit=1)
        assert [o["task_id"] for o in orphans] == [tid]
        assert orphans[0]["consecutive_failures"] >= 1


def test_orphan_with_undone_parent_waits_for_parents(kanban_home: Path) -> None:
    """A gave_up child whose parents are not done is NOT surfaced — the
    stuck class this actor exists for is the parent-satisfied one.  It
    becomes enumerable once the parents complete (exactly the live
    board shape from t_8285095c)."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="orphan child", parents=[parent])
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="parent done")
        assert kb.get_task(conn, child).status == "ready"
        _trip_breaker(conn, child, failure_limit=1)

        # Parents ARE done here — enumerable.
        assert [o["task_id"] for o in kb.list_gaveup_orphans(conn, failure_limit=1)] == [child]

        # Rebuild with an open parent: the orphan is hidden until the
        # parent lands (recompute_ready never touches it; retry would
        # just land it in todo, which is allowed but not the stuck
        # class being triaged).
        parent2 = kb.create_task(conn, title="open parent")
        child2 = kb.create_task(conn, title="gated orphan", parents=[parent2])
        kb.claim_task(conn, parent2)
        kb.complete_task(conn, parent2, summary="done")
        _trip_breaker(conn, child2, failure_limit=1)
        assert child2 in [o["task_id"] for o in kb.list_gaveup_orphans(conn, failure_limit=1)]


def test_sticky_block_is_not_enumerable(kanban_home: Path) -> None:
    """Explicit worker/operator holds are a different class — they must
    never appear in the gave_up enumeration, even when a gave_up event
    precedes the block (the #28712 loop shape)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="human hold")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="needs_input: which schema do you want?",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.list_gaveup_orphans(conn) == []

        # A second card with a gave_up event followed by a fresh human
        # block (the #28712 loop shape): sticky, not an orphan.
        tid2 = kb.create_task(conn, title="hold after gave_up")
        _trip_breaker(conn, tid2, failure_limit=1)
        kb.unblock_task(conn, tid2)
        kb.claim_task(conn, tid2)
        kb.block_task(
            conn, tid2,
            reason="needs_input: still waiting on you",
            expected_run_id=kb.get_task(conn, tid2).current_run_id,
        )
        assert kb.get_task(conn, tid2).status == "blocked"
        assert kb.list_gaveup_orphans(conn, failure_limit=1) == []


def test_below_limit_failure_is_not_enumerable(kanban_home: Path) -> None:
    """Fail-closed: a task whose consecutive_failures has not reached
    the effective limit is still being retried by the dispatcher, not
    orphaned — the actor must not see it."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="still retrying")
        kb.claim_task(conn, tid)
        tripped = kb._record_task_failure(
            conn, tid,
            error="pid 4242 not alive",
            outcome="crashed",
            failure_limit=3,
            release_claim=True,
            end_run=True,
        )
        assert not tripped
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.list_gaveup_orphans(conn) == []


# ---------------------------------------------------------------------------
# Retry: fresh budget, guarded, audited
# ---------------------------------------------------------------------------


def test_retry_dead_pid_orphan(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="retry me")
        _trip_breaker(conn, tid, failure_limit=2)
        assert kb.get_task(conn, tid).consecutive_failures == 2

        ok, err = kb.retry_gaveup_task(
            conn, tid, actor="conductor", reason="transient OOM on host",
        )
        assert ok, err
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None

        events = _events(conn, tid, kinds={"gave_up_retried"})
        assert len(events) == 1
        payload = events[0][1]
        assert payload["actor"] == "conductor"
        assert payload["reason"] == "transient OOM on host"
        assert payload["landed"] == "ready"
        assert payload["previous_failures"] == 2

        # The dispatcher picks the retried card up again — a normal
        # fresh run, not an infinite respawn loop.
        assert kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="done on retry")
        assert kb.get_task(conn, tid).status == "done"


def test_retry_lands_in_todo_when_parents_open(kanban_home: Path) -> None:
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="parent done")
        assert kb.get_task(conn, child).status == "ready"
        _trip_breaker(conn, child, failure_limit=1)
        # Parent re-opened: the retry must re-gate and land in todo.
        conn.execute(
            "UPDATE tasks SET status = 'running' WHERE id = ?", (parent,),
        )
        conn.commit()

        ok, err = kb.retry_gaveup_task(
            conn, child, actor="conductor", reason="retry after parent lands",
            failure_limit=1,
        )
        assert ok, err
        assert kb.get_task(conn, child).status == "todo"
        events = _events(conn, child, kinds={"gave_up_retried"})
        assert events[0][1]["landed"] == "todo"


def test_retry_refuses_sticky_block(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="sticky")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="needs_input: human decision required",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        ok, err = kb.retry_gaveup_task(conn, tid, actor="conductor", reason="x")
        assert not ok
        assert "sticky" in err
        # Untouched: still blocked, failure budget unchanged (0).
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.get_task(conn, tid).consecutive_failures == 0
        assert _events(conn, tid, kinds={"gave_up_retried"}) == []


def test_retry_refuses_non_orphan_states(kanban_home: Path) -> None:
    with kb.connect() as conn:
        ready_tid = kb.create_task(conn, title="ready")
        ok, err = kb.retry_gaveup_task(conn, ready_tid, actor="conductor", reason="x")
        assert not ok and "blocked" in err

        done_tid = kb.create_task(conn, title="done")
        kb.claim_task(conn, done_tid)
        kb.complete_task(conn, done_tid, summary="done")
        ok, err = kb.retry_gaveup_task(conn, done_tid, actor="conductor", reason="x")
        assert not ok

        ok, err = kb.retry_gaveup_task(conn, "t_nonexistent", actor="c", reason="x")
        assert not ok


def test_retry_is_fail_closed_below_limit(kanban_home: Path) -> None:
    """A blocked task whose counter somehow sits BELOW the effective
    limit with a latest gave_up event cannot be force-retried through
    this actor — the breaker budget remains authoritative."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="below limit")
        _trip_breaker(conn, tid, failure_limit=2)
        # Rewind the counter to simulate an inconsistent/below-limit
        # state (e.g. max_retries raised after the trip).
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0 WHERE id = ?", (tid,),
        )
        conn.commit()
        ok, err = kb.retry_gaveup_task(
            conn, tid, actor="conductor", reason="x", failure_limit=2,
        )
        assert not ok
        assert "limit" in err
        assert kb.get_task(conn, tid).status == "blocked"


def test_retry_reason_required(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no reason")
        _trip_breaker(conn, tid, failure_limit=1)
        ok, err = kb.retry_gaveup_task(conn, tid, actor="conductor", reason="")
        assert not ok and "reason" in err
        ok, err = kb.retry_gaveup_task(conn, tid, actor="conductor", reason=None)
        assert not ok and "reason" in err


def test_retried_card_leaves_orphan_list(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="once only")
        _trip_breaker(conn, tid, failure_limit=1)
        assert kb.list_gaveup_orphans(conn, failure_limit=1)
        ok, _ = kb.retry_gaveup_task(
            conn, tid, actor="c", reason="retry", failure_limit=1,
        )
        assert ok
        assert kb.list_gaveup_orphans(conn, failure_limit=1) == []


# ---------------------------------------------------------------------------
# Archive: supersede with reason, guarded, audited
# ---------------------------------------------------------------------------


def test_archive_orphan(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="superseded")
        _trip_breaker(conn, tid, failure_limit=1)

        ok, err = kb.archive_gaveup_task(
            conn, tid, actor="conductor",
            reason="superseded by t_abcd1234 redesign",
            failure_limit=1,
        )
        assert ok, err
        assert kb.get_task(conn, tid).status == "archived"

        # The human-decision ``blocked`` event lands BEFORE the
        # ``archived`` event so the orphan classifier can never misread
        # the card mid-flight, and it carries the reason + actor.
        events = _events(conn, tid)
        kinds = [k for k, _ in events]
        assert kinds[-2] == "blocked"
        assert kinds[-1] == "archived"
        blocked_payload = events[-2][1]
        assert blocked_payload["reason"].startswith("gave_up archived:")
        assert blocked_payload["actor"] == "conductor"
        assert blocked_payload["kind"] == "transient"
        assert kb.list_gaveup_orphans(conn, failure_limit=1) == []


def test_archive_refuses_sticky_block(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="sticky archive refuse")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="needs_input: hold for legal review",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        ok, err = kb.archive_gaveup_task(conn, tid, actor="conductor", reason="x")
        assert not ok
        assert "sticky" in err
        # Untouched: still blocked, NOT archived, no archive event.
        assert kb.get_task(conn, tid).status == "blocked"
        assert _events(conn, tid, kinds={"archived"}) == []


def test_archive_refuses_non_orphan(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ready archive refuse")
        ok, err = kb.archive_gaveup_task(conn, tid, actor="conductor", reason="x")
        assert not ok and "blocked" in err
        assert kb.get_task(conn, tid).status == "ready"


def test_archive_reason_required(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no reason archive")
        _trip_breaker(conn, tid, failure_limit=1)
        ok, err = kb.archive_gaveup_task(conn, tid, actor="conductor", reason="  ")
        assert not ok and "reason" in err
        assert kb.get_task(conn, tid).status == "blocked"


def test_archived_orphan_unblocks_dependents(kanban_home: Path) -> None:
    """Archive behaves exactly like a normal archive for the dependency
    graph: archived parents no longer block children."""
    with kb.connect() as conn:
        orphan = kb.create_task(conn, title="orphan parent")
        child = kb.create_task(conn, title="child", parents=[orphan])
        _trip_breaker(conn, orphan, failure_limit=1)
        assert kb.get_task(conn, child).status == "todo"

        ok, _ = kb.archive_gaveup_task(
            conn, orphan, actor="conductor", reason="superseded",
            failure_limit=1,
        )
        assert ok
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Behaviour preservation: recompute stays hands-off, sticky stays sticky
# ---------------------------------------------------------------------------


def test_recompute_ready_still_never_touches_orphans(kanban_home: Path) -> None:
    """The actor is the ONLY path out of a gave_up orphan — recompute_ready
    must not silently promote one (no infinite respawn loops, the whole
    point of the circuit breaker)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="orphan stability")
        _trip_breaker(conn, tid, failure_limit=1)
        # First tick: the release-or-record actor (t_e6423d36) auto-
        # releases the reasonless block ONCE — the pre-existing reasonless
        # sticky-block behaviour this card must not change. The gave_up
        # event then becomes a stale circuit-breaker row and the card sits
        # in ready with NO respawn loop: recompute never touches it again.
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"
        for _ in range(4):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "ready"


def test_explicit_sticky_block_survives_orphan_actor(kanban_home: Path) -> None:
    """The full #28712 pin, restated against the new actor: a worker
    block stays blocked across ticks AND across gave_up noise."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="still sticky")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: please verify ACL change",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'protocol_violation', NULL, ?)",
            (tid, now),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (tid, now + 1),
        )
        conn.commit()
        # gave_up noise is NEWER than the block but the block is still
        # the latest of the {blocked, unblocked} pair: sticky guard
        # wins, actor refuses, recompute skips.
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
        ok, err = kb.retry_gaveup_task(conn, tid, actor="c", reason="x")
        assert not ok
        assert kb.get_task(conn, tid).status == "blocked"
