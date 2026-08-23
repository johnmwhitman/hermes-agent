"""Tests for the kanban_db DEFAULT_MIN_WORKER_RUNTIME_SECONDS gate.

Added 2026-08-23 to close t_d3582171 (kanban_db.py fix absent from all 3
codebases, found 2026-08-22). The gate is the dispatcher-side counterpart
to the cron-hollow guard: a worker that exits rc=0 with no result/summary
in under DEFAULT_MIN_WORKER_RUNTIME_SECONDS is treated as HOLLOW, left in
its prior status, and audited via completion_blocked_short_runtime event.
"""
import os
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Hermetic HERMES_HOME + kanban DB; resets the singleton between tests."""
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    (hermes / "kanban").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.delenv("HERMES_KANBAN_MIN_WORKER_RUNTIME_SECONDS", raising=False)
    # Force the connect() singleton to re-resolve against the new HERMES_HOME
    if hasattr(kb, "_KANBAN_DB"):
        kb._KANBAN_DB = None
    yield hermes


def test_default_min_worker_runtime_seconds_constant_exists():
    assert hasattr(kb, "DEFAULT_MIN_WORKER_RUNTIME_SECONDS")
    assert isinstance(kb.DEFAULT_MIN_WORKER_RUNTIME_SECONDS, int)
    assert kb.DEFAULT_MIN_WORKER_RUNTIME_SECONDS > 0


def test_resolve_helper_returns_positive_int():
    val = kb._resolve_min_worker_runtime_seconds()
    assert isinstance(val, int)
    assert val > 0


def test_resolve_helper_honors_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_MIN_WORKER_RUNTIME_SECONDS", "12")
    assert kb._resolve_min_worker_runtime_seconds() == 12


def test_resolve_helper_zero_disables_gate(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_MIN_WORKER_RUNTIME_SECONDS", "0")
    assert kb._resolve_min_worker_runtime_seconds() == 0


def test_complete_task_rejects_short_runtime_with_empty_payload(kanban_home):
    """A worker that exits rc=0 in <DEFAULT_MIN_WORKER_RUNTIME_SECONDS
    with no result and no summary should be blocked, not completed."""
    db_path = kanban_home / "kanban" / "kanban.db"
    conn = kb.connect(db_path)
    tid = kb.create_task(conn, title="hollow-worker-test")
    kb.claim_task(conn, tid)
    now = int(time.time())
    # NOTE: real schema is task_runs (NOT runs). Use the correct table.
    conn.execute(
        "INSERT INTO task_runs(task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        (tid, "done", now - 5, now, "completed"),
    )
    conn.commit()
    ok = kb.complete_task(conn, tid)
    assert not ok, "complete_task should refuse hollow completion"
    task = kb.get_task(conn, tid)
    assert task is not None and task.status != "done", \
        "task status must not flip to done on HOLLOW"
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? "
        "AND kind = 'completion_blocked_short_runtime'",
        (tid,),
    ).fetchall()
    assert len(events) == 1


def test_complete_task_allows_long_runtime_with_empty_payload(kanban_home):
    """A worker that ran >DEFAULT_MIN_WORKER_RUNTIME_SECONDS is fine even
    with empty result/summary — long runtime implies real work."""
    db_path = kanban_home / "kanban" / "kanban.db"
    conn = kb.connect(db_path)
    tid = kb.create_task(conn, title="long-runtime-worker")
    kb.claim_task(conn, tid)
    now = int(time.time())
    conn.execute(
        "INSERT INTO task_runs(task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        (tid, "done", now - 120, now, "completed"),
    )
    conn.commit()
    assert kb.complete_task(conn, tid) is True


def test_complete_task_allows_short_runtime_with_result(kanban_home):
    """A short-runtime worker with a real result is fine — the gate only
    blocks the (short-runtime AND empty-payload) intersection."""
    db_path = kanban_home / "kanban" / "kanban.db"
    conn = kb.connect(db_path)
    tid = kb.create_task(conn, title="short-runtime-with-result")
    kb.claim_task(conn, tid)
    now = int(time.time())
    conn.execute(
        "INSERT INTO task_runs(task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        (tid, "done", now - 2, now, "completed"),
    )
    conn.commit()
    assert kb.complete_task(conn, tid, result="real work done") is True


def test_does_not_use_runs_table():
    """Regression: the fix must query task_runs, never the (non-existent) runs table."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(repo, "hermes_cli", "kanban_db.py")).read()
    # The buggy arm queried `FROM runs WHERE task_id = ?` — a table that
    # doesn't exist in the schema. Verify it's gone.
    assert "FROM runs WHERE" not in src, \
        "buggy `FROM runs` query is still in source"
    assert "FROM task_runs WHERE" in src, \
        "fixed `FROM task_runs` query is missing"
