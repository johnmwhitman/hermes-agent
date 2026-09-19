"""Regression tests for the block-loop circuit breaker (t_6374a351).

The forensic finding was that ``block_recurrences`` only incremented when
``prev_kind == kind`` AND the new block's ``reason`` text matched. The
Overwatch-guardian-verification card family (t_ab9f0f2e, t_80e0a9c1,
t_d5dab1b2, t_0bea810b, t_7157edab) defeated the breaker by varying the
block reason string each cycle (different finding name / different report
timestamp), and the column counter stayed pinned at 1 while
``block_loop_detected`` fired 60x on t_ab9f0f2e alone. The fix removes
the free-text reason match and the ``prev_kind == kind`` reset: every
re-block of the same task now increments ``block_recurrences`` and routes
to ``triage`` at ``BLOCK_RECURRENCE_LIMIT`` regardless of reason-text
variation or ``retype_block_kind`` rewrites.

These tests pin the new behaviour so a future regression cannot restore
the silent zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    """Reset the card back to a claimable state so block_task can re-block."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _block_cycle(conn, tid, *, reason, kind="capability"):
    """block -> unblock -> claim cycle so the next block_task is on a running card."""
    assert kb.block_task(conn, tid, reason=reason, kind=kind)
    assert kb.unblock_task(conn, tid)
    _make_running_again(conn, tid)


# ---------------------------------------------------------------------------
# Regression: varying reason text must NOT reset block_recurrences
# ---------------------------------------------------------------------------


def test_varying_reason_text_increments_recurrences(kanban_home: Path) -> None:
    """The exact bypass: block N times with a DIFFERENT reason string each
    cycle (mirrors the guardian-verification template). The counter must
    climb every time and reach ``BLOCK_RECURRENCE_LIMIT`` (=2), routing the
    card to ``triage`` on the second block-loop event. Pre-fix the counter
    reset to 1 on every cycle because prev_kind==kind was the only path
    that incremented; the new logic increments unconditionally.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="varying-reason")
        # First block lands recurrences=1, status=blocked.
        assert kb.block_task(
            conn, tid, reason="finding X report ts=2026-09-14T14Z", kind="capability",
        )
        first = kb.get_task(conn, tid)
        assert first is not None
        assert first.status == "blocked"
        assert first.block_recurrences == 1
        assert first.block_kind == "capability"

        # First unblock+reblock with a CHANGED reason string but the SAME kind.
        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        assert kb.block_task(
            conn, tid, reason="finding X report ts=2026-09-15T14Z (rewritten)", kind="capability",
        )
        # Pre-fix the counter reset to 1 here because prev_kind==kind held but
        # the implementation compared kind, not reason; with kind unchanged
        # this path used to actually increment. The real bypass path is the
        # kind-change test below; this test still pins that the new payload
        # shape carries ``prev_kind`` + ``kind_changed`` for forensics.
        second = kb.get_task(conn, tid)
        assert second is not None
        assert second.block_recurrences == 2, (
            f"second block with same kind + changed reason must reach "
            f"BLOCK_RECURRENCE_LIMIT (recurrences={second.block_recurrences})"
        )
        assert second.status == "triage"
        events = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
        assert events, "expected block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"
        assert payload.get("prev_kind") == "capability"
        assert payload.get("kind_changed") is False


def test_changing_kind_does_not_reset_recurrences(kanban_home: Path) -> None:
    """The retype_block_kind bypass: stored ``block_kind`` differs from the
    incoming ``kind`` (because a prior cycle ran ``retype_block_kind`` and
    mutated the column without updating ``block_recurrences``). Pre-fix
    this caused ``prev_kind != kind`` and reset the counter to 1 every
    cycle. The new behaviour increments regardless.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="kind-change")
        # First block with kind=capability.
        assert kb.block_task(conn, tid, reason="initial", kind="capability")
        first = kb.get_task(conn, tid)
        assert first is not None
        assert first.block_recurrences == 1
        assert first.block_kind == "capability"

        # Re-type the stored block_kind to a different value (mirrors what
        # the operator did on t_ab9f0f2e when audit retyped transient->capability).
        ok, _ = kb.retype_block_kind(conn, tid, kind="needs_input", reason="audit retype")
        assert ok
        ret = kb.get_task(conn, tid)
        assert ret is not None
        assert ret.block_kind == "needs_input"
        # Retype must NOT change the recurrences counter — the audit repair
        # is a metadata correction, not a re-block.
        assert ret.block_recurrences == 1

        # Unblock, claim again, then re-block with the WORKER's kind
        # (which is still capability in the real Overwatch scenario).
        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        assert kb.block_task(conn, tid, reason="next finding", kind="capability")

        # Pre-fix: prev_kind == 'needs_input', new kind == 'capability',
        # prev_kind != kind -> recurrences reset to 1.
        # Post-fix: counter always increments, so we land at 2 -> triage.
        second = kb.get_task(conn, tid)
        assert second is not None
        assert second.block_recurrences == 2, (
            f"kind-change re-block must still hit BLOCK_RECURRENCE_LIMIT; "
            f"got recurrences={second.block_recurrences}"
        )
        assert second.status == "triage"
        events = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
        payload = events[-1].payload or {}
        assert payload.get("kind_changed") is True
        assert payload.get("prev_kind") == "needs_input"
        assert payload.get("kind") == "capability"


def test_pure_reason_text_variation_breaks_loop(kanban_home: Path) -> None:
    """The exact reproduction from the forensic finding: block the same
    task repeatedly with a DIFFERENT reason string each time (different
    finding name / report timestamp). Pre-fix the counter was pinned at 1
    forever and the card stayed in ``blocked`` forever. Post-fix the
    counter climbs monotonically and the card lands in ``triage`` at
    ``BLOCK_RECURRENCE_LIMIT`` (=2) for human review, then refuses further
    block/unblock cycles as designed.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="guardian-mirror")
        # Cycle 1: first block lands in 'blocked' with recurrences=1.
        assert kb.block_task(
            conn, tid, reason="finding repo.history:machine ts=2026-09-14T14:19Z.md",
            kind="transient",
        )
        first = kb.get_task(conn, tid)
        assert first is not None
        assert first.block_recurrences == 1
        assert first.status == "blocked"

        # Cycle 2: unblock + re-block with a DIFFERENT reason string.
        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        assert kb.block_task(
            conn, tid, reason="finding repo.history:machine ts=2026-09-15T14:19Z.md (re-listed)",
            kind="transient",
        )
        # Pre-fix: counter reset to 1 because prev_kind==kind was the only
        # increment path AND the implementation ignored reason-text variation.
        # Post-fix: counter increments to 2, status flips to 'triage', the
        # loop breaker fires.
        second = kb.get_task(conn, tid)
        assert second is not None
        assert second.block_recurrences == 2, (
            f"counter must climb across the second re-block despite reason-text "
            f"variation; got recurrences={second.block_recurrences} (pinned=1 = bug back)"
        )
        assert second.status == "triage"

        # Cycle 3: once in triage, block_task is a no-op (the card left the
        # human-blocked bucket). This is the design — triage is for human
        # review, not for further block/unblock cycles. The breaker has fired.
        assert kb.unblock_task(conn, tid) is False, (
            "unblock on a triage-status card must refuse; the loop breaker "
            "has already escalated and the card is awaiting human review"
        )
        events = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event to land in triage"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("limit") == kb.BLOCK_RECURRENCE_LIMIT


def test_unblock_does_not_reset_recurrences(kanban_home: Path) -> None:
    """Defensive: confirm the unblock path still preserves the counter (the
    comment on unblock_task says so). With the new always-increment rule a
    regression that reset the counter on unblock would erase the loop
    signal entirely.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="preserve-counter")
        kb.block_task(conn, tid, reason="r1", kind="capability")
        kb.unblock_task(conn, tid)
        # Column state should still hold the post-first-block values.
        mid = kb.get_task(conn, tid)
        assert mid is not None
        assert mid.status != "blocked"
        assert mid.block_recurrences == 1, (
            "unblock must preserve block_recurrences; resetting here is the "
            "amnesia the new logic explicitly defends against"
        )
        assert mid.block_kind == "capability"


def test_complete_task_clears_recurrences(kanban_home: Path) -> None:
    """Defensive: completion still resets the counter so a genuinely
    resolved task starts fresh on its next run. (No production change here;
    this pins the existing reset-on-complete behaviour.)
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="complete-clears")
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="y", kind="capability")
        # 2nd re-block already routes to triage; that's fine — we just want
        # to assert completion still wipes the counter on a separate task.
        done = kb.get_task(conn, tid)
        assert done is not None
        assert done.block_recurrences >= 2
        # A separate task to test the completion reset cleanly:
        tid2 = _running_task(conn, title="clean-complete")
        kb.block_task(conn, tid2, reason="z", kind="capability")
        # Mark complete while in blocked — but complete_task needs the task
        # back to ready/running, so unblock first.
        kb.unblock_task(conn, tid2)
        _make_running_again(conn, tid2)
        kb.complete_task(conn, tid2, result="done")
        cleared = kb.get_task(conn, tid2)
        assert cleared is not None
        assert cleared.block_recurrences == 0
        assert cleared.status == "done"
