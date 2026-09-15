"""Tests for the block_kind tying mechanism (t_19c3e315).

Before this fix, two surfaces could mint a card with ``status=blocked``
but ``block_kind=NULL``:

  * ``hermes kanban create --initial-status blocked`` -- no --kind option
    existed on create, so the kind was always NULL.
  * No supported command could later set block_kind on an already-blocked
    card (``block_task`` refuses because status is already 'blocked').

This file verifies the two mechanisms that close the gap:

  1. ``create_task(initial_status='blocked', initial_block_kind=...)``
     stamps block_kind into the new row in one atomic INSERT.
  2. ``retype_block_kind`` repairs an already-blocked card WITHOUT
     rewriting status/claim_lock/recurrences or emitting a fresh
     ``blocked`` transition event.  It refuses on non-blocked tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Path A: create_task initial_block_kind
# ---------------------------------------------------------------------------


def test_create_blocked_with_kind_stamps_block_kind(kanban_home):
    """create_task(initial_status='blocked', initial_block_kind=...)
    writes the kind on the new row in one transaction."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="human-ops card",
            assignee="conductor",
            initial_status="blocked",
            initial_block_kind="needs_input",
        )
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"


def test_create_blocked_without_kind_stamps_default_needs_input(kanban_home):
    """Initial-status=blocked without a kind used to land as NULL block_kind,
    which the audit query that filters by kind could not see (24 NULL rows
    in production). The DB layer now stamps ``needs_input`` as a default so
    fleet-health bridges, goal-mode judges, and dashboard creation all
    produce a typed row. The CLI arg parser additionally enforces --kind;
    this is the safety net for programmatic callers.
    """
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="default-kind card",
            assignee="conductor",
            initial_status="blocked",
        )
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"


def test_create_with_kind_but_not_blocked_raises(kanban_home):
    """block_kind only makes sense with initial_status='blocked'. The DB
    layer mirrors the CLI parser contract so the failure is loud and
    immediate rather than silently dropping the value."""
    with pytest.raises(ValueError, match="initial_block_kind"):
        with kb.connect_closing() as conn:
            kb.create_task(
                conn,
                title="misuse",
                assignee="conductor",
                initial_status="running",
                initial_block_kind="needs_input",
            )


def test_create_with_invalid_kind_raises(kanban_home):
    with pytest.raises(ValueError, match="initial_block_kind"):
        with kb.connect_closing() as conn:
            kb.create_task(
                conn,
                title="bad kind",
                assignee="conductor",
                initial_status="blocked",
                initial_block_kind="bogus_kind",
            )


def test_create_blocked_event_payload_records_kind(kanban_home):
    """Audit trail must surface the kind alongside the existing
    ``{"initial": True}`` marker so a future investigator can read the
    reason from task_events without joining against a side comment."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="audited",
            assignee="conductor",
            initial_status="blocked",
            initial_block_kind="capability",
        )
        ev = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'blocked' "
            "ORDER BY id ASC LIMIT 1",
            (tid,),
        ).fetchone()
    payload = json.loads(ev["payload"])
    assert payload.get("initial") is True
    assert payload.get("kind") == "capability"


# ---------------------------------------------------------------------------
# Path B: retype_block_kind on already-blocked cards
# ---------------------------------------------------------------------------


def _make_blocked(conn, *, block_kind=None):
    """Helper to mint a card in status=blocked. When ``block_kind`` is None
    the helper bypasses the create_task safety net (which now stamps
    ``needs_input`` by default) by writing NULL directly, so legacy
    stragglers can still be simulated for retype tests.
    """
    tid = kb.create_task(
        conn,
        title="legacy blocked",
        assignee="conductor",
        initial_status="blocked",
        initial_block_kind=block_kind,  # may be None for legacy
    )
    if block_kind is None:
        conn.execute(
            "UPDATE tasks SET block_kind = NULL WHERE id = ?", (tid,)
        )
    return tid


def test_retype_sets_block_kind_on_untyped_blocked_card(kanban_home):
    with kb.connect_closing() as conn:
        tid = _make_blocked(conn)  # NULL block_kind, like the 68 stragglers

        ok, why = kb.retype_block_kind(
            conn, tid, kind="needs_input", reason="CoS measurement t_19c3e315",
        )
        assert ok is True
        assert why is None

        row = conn.execute(
            "SELECT status, block_kind, block_recurrences, claim_lock "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
    assert row["block_kind"] == "needs_input"
    # The repair MUST NOT mutate the lifecycle state. Retyping is a
    # metadata correction, not a transition.
    assert row["status"] == "blocked"
    assert row["block_recurrences"] == 0
    assert row["claim_lock"] is None


def test_retype_emits_audit_event_without_blocked_transition(kanban_home):
    with kb.connect_closing() as conn:
        tid = _make_blocked(conn)
        ok, _ = kb.retype_block_kind(
            conn, tid, kind="needs_input", reason="audit test",
        )
        assert ok is True
        kinds = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? "
                "ORDER BY id ASC",
                (tid,),
            ).fetchall()
        ]
    # created + blocked(initial) + block_kind_retyped. NO fresh 'blocked'
    # transition -- that's the whole point of the retype path.
    assert kinds == ["created", "blocked", "block_kind_retyped"]


def test_retype_refuses_on_non_blocked_task(kanban_home):
    """The helper MUST refuse if status != 'blocked', so a worker can't
    use it to set kind on a card mid-flight and confuse the dispatcher's
    block-loop breaker (recurrences only increment on real transitions)."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ready card", assignee="conductor")
        ok, why = kb.retype_block_kind(
            conn, tid, kind="needs_input", reason="should refuse",
        )
    assert ok is False
    assert "not 'blocked'" in (why or "")


def test_retype_refuses_when_kind_already_matches(kanban_home):
    with kb.connect_closing() as conn:
        tid = _make_blocked(conn, block_kind="capability")
        ok, why = kb.retype_block_kind(
            conn, tid, kind="capability", reason="no-op",
        )
    assert ok is False
    assert "already has block_kind" in (why or "")


def test_retype_requires_kind():
    """The bare helper requires a kind to retype to."""
    with kb.connect_closing() as conn:
        tid = _make_blocked(conn)
        ok, why = kb.retype_block_kind(conn, tid, kind=None, reason="x")
    assert ok is False
    assert "--kind" in (why or "")


def test_retype_requires_reason():
    """The bare helper requires a reason for the audit trail."""
    with kb.connect_closing() as conn:
        tid = _make_blocked(conn)
        ok_no_reason, why_no_reason = kb.retype_block_kind(
            conn, tid, kind="needs_input", reason=None,
        )
        ok_blank, why_blank = kb.retype_block_kind(
            conn, tid, kind="needs_input", reason="   ",
        )
    assert ok_no_reason is False and "reason" in (why_no_reason or "").lower()
    assert ok_blank is False and "reason" in (why_blank or "").lower()


def test_retype_invalid_kind():
    with kb.connect_closing() as conn:
        tid = _make_blocked(conn)
        ok, why = kb.retype_block_kind(
            conn, tid, kind="garbage", reason="x",
        )
    assert ok is False
    assert "invalid kind" in (why or "")


# ---------------------------------------------------------------------------
# CLI surface: --kind on create, --retype on block
# ---------------------------------------------------------------------------


def test_cli_create_blocked_without_kind_returns_nonzero(kanban_home):
    """The CLI parser enforces the new contract: --initial-status
    blocked requires --kind. This is the user-visible half of Path A."""
    from hermes_cli import kanban as kc

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "create", "human-ops",
         "--initial-status", "blocked",
         "--assignee", "conductor"]
    )
    rc = kc.kanban_command(args)
    assert rc == 2


def test_cli_create_blocked_with_kind_succeeds(kanban_home):
    from hermes_cli import kanban as kc

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "create", "human-ops",
         "--initial-status", "blocked",
         "--kind", "needs_input",
         "--assignee", "conductor"]
    )
    rc = kc.kanban_command(args)
    assert rc == 0
    # Verify the kind landed.
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE title = 'human-ops'"
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"


def test_cli_block_retype_repairs_untyped_card(kanban_home):
    """End-to-end: create a card via the legacy NULL-kind path (simulating
    older DB rows that pre-date the schema-tightening patch), then
    `hermes kanban block --retype --kind` repairs it without rewriting
    status history. The fix is to backfill block_kind on legacy NULL rows
    in production (see _record_task_failure / block_task fallbacks).
    """
    from hermes_cli import kanban as kc

    # Mint a NULL-kind card directly via the DB layer (the CLI parser
    # now refuses this path; we test the legacy data that already
    # exists on the board).
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="legacy", assignee="conductor",
            initial_status="blocked",
        )
        # The new create_task safety net stamps needs_input; force it back
        # to NULL to simulate a row that pre-dates the safety net.
        conn.execute(
            "UPDATE tasks SET block_kind = NULL WHERE id = ?", (tid,)
        )

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "block", "--retype",
         "--kind", "needs_input",
         tid,
         "CoS measurement t_19c3e315"]
    )
    rc = kc.kanban_command(args)
    assert rc == 0

    with kb.connect() as conn:
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"


def test_cli_block_retype_requires_reason(kanban_home, capsys):
    """--retype without a reason is rejected loudly so audit trails
    never go silent."""
    from hermes_cli import kanban as kc

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="silent retype", assignee="conductor",
            initial_status="blocked",
        )

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "block", "--retype", "--kind", "needs_input", tid]
    )
    rc = kc.kanban_command(args)
    captured = capsys.readouterr()
    assert rc == 2
    assert "reason" in captured.err.lower()


def test_cli_block_retype_requires_kind(kanban_home, capsys):
    from hermes_cli import kanban as kc

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="retype no kind", assignee="conductor",
            initial_status="blocked",
        )

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "block", "--retype", tid, "needs a reason"]
    )
    rc = kc.kanban_command(args)
    captured = capsys.readouterr()
    assert rc == 2
    assert "--kind" in captured.err


# ---------------------------------------------------------------------------
# Schema-tightening fallbacks (t_3339b5b2):
#   * block_task(kind=None) used to leave block_kind NULL; the goal-mode
#     judge that ruled t_39a31cf3 unachievable hit this path.
#   * _record_task_failure() used to flip status='blocked' but never
#     stamped block_kind; 13 of the 17 NULL-block_kind rows on production
#     came from there (crash / timeout / protocol-violation gave_up paths).
#   * create_task(initial_status='blocked', initial_block_kind=None) used
#     to leave block_kind NULL; 3 production rows came from the legacy path
#     (fleet-health bridge, dashboard, programmatic).
# All three fallbacks now stamp a default kind so the audit query that
# filters by kind sees every blocked card.
# ---------------------------------------------------------------------------


def test_block_task_without_kind_stamps_needs_input(kanban_home):
    """block_task(kind=None) (the goal-mode judge path) lands with
    block_kind='needs_input' instead of NULL.
    """
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="to-be-blocked", assignee="conductor",
        )
        ok = kb.block_task(conn, tid, reason="judge gave up", kind=None)
        assert ok is True
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"


def test_block_task_with_explicit_kind_preserves_it(kanban_home):
    """The fallback must not overwrite an explicit kind."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="to-be-blocked-typed", assignee="conductor",
        )
        ok = kb.block_task(
            conn, tid, reason="capability gate", kind="capability",
        )
        assert ok is True
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "capability"


def test_record_task_failure_stamps_transient(kanban_home):
    """The dispatcher's failure breaker trips to status='blocked' but used
    to leave block_kind NULL. The 13 NULL-kind gave_up rows on production
    came from this path. Now stamps ``transient`` (the breaker-tripping
    flavor — same vocabulary ``kanban_block(kind='transient')`` uses).
    """
    from hermes_cli.kanban_db_dispatch import _record_task_failure

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="crashing-task", assignee="conductor",
        )
        # Claim it (running + claim_lock) so release_claim semantics match.
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock='test', "
            "current_run_id=1 WHERE id = ?", (tid,)
        )
        tripped = _record_task_failure(
            conn, tid,
            error="pid 999 not alive without a terminal kanban call",
            outcome="crashed",
            failure_limit=1,
            release_claim=True,
            end_run=True,
        )
        assert tripped is True
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "transient"
