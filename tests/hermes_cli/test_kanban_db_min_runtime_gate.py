#!/usr/bin/env python3
"""Tests for the kanban_db DEFAULT_MIN_WORKER_RUNTIME_SECONDS gate.

Closes t_d3582171 (researcher false-completion bug, 3rd occurrence 2026-08-23):
workers exit rc=0 with no result/summary in <30s without doing real work. The
dispatcher must hold the task at its prior status and emit a HOLLOW signal.

Three cases:
  1. Worker exits rc=0 with empty result/summary in <30s → HOLLOW, NOT marked done.
  2. Worker exits rc=0 with empty result/summary in >30s → marked done.
  3. Worker exits rc=0 with a result OR summary → marked done regardless of runtime.

Plus the env-var override and a regression check that the buggy `runs` table
name is gone (we use `task_runs`).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# Add the hermes-agent checkout to sys.path so the test can import kanban_db
REPO = Path("/Users/johnwhitman/.hermes/hermes-agent")
sys.path.insert(0, str(REPO))


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """A scratch HERMES_HOME with a kanban DB that has the real task_runs schema."""
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    (hermes / "kanban").mkdir()
    db_path = hermes / "kanban" / "kanban.db"
    con = sqlite3.connect(db_path)
    con.executescript(open(REPO / "hermes_cli" / "schema.sql").read())
    con.close()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.delenv("HERMES_KANBAN_MIN_WORKER_RUNTIME_SECONDS", raising=False)
    return hermes, db_path


def _insert_task(con, task_id: str = "t_test", status: str = "running"):
    now = int(time.time())
    con.execute(
        "INSERT INTO tasks (id, title, status, created_at, started_at, idempotency_key) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, f"task {task_id}", status, now, now, f"k-{task_id}"),
    )
    con.commit()


def _insert_run(con, task_id: str, started_at: int, ended_at: int | None = None,
                outcome: str = "completed"):
    con.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, "done", started_at, ended_at, outcome),
    )
    con.commit()


def test_hollow_short_run_no_result_blocks_completion(tmp_hermes_home):
    """Worker exits rc=0 in <30s with no result/summary → HOLLOW, return False."""
    hermes, db_path = tmp_hermes_home
    from hermes_cli import kanban_db
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    now = int(time.time())
    _insert_task(con)
    _insert_run(con, "t_test", started_at=now - 5, ended_at=now)
    con.close()

    # complete_task with empty summary/result should be rejected
    result = kanban_db.complete_task(
        task_id="t_test",
        summary=None,
        result=None,
    )
    assert result is False, "Expected HOLLOW rejection; got success"

    con = sqlite3.connect(db_path)
    row = con.execute("SELECT status FROM tasks WHERE id='t_test'").fetchone()
    assert row[0] == "running", f"expected status=running, got {row[0]}"
    con.close()


def test_long_run_no_result_completes(tmp_hermes_home):
    """Worker exits rc=0 in >30s with no result/summary → marked done."""
    hermes, db_path = tmp_hermes_home
    from hermes_cli import kanban_db
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    now = int(time.time())
    _insert_task(con)
    _insert_run(con, "t_test", started_at=now - 60, ended_at=now)
    con.close()

    result = kanban_db.complete_task(
        task_id="t_test",
        summary=None,
        result=None,
    )
    assert result is True

    con = sqlite3.connect(db_path)
    row = con.execute("SELECT status FROM tasks WHERE id='t_test'").fetchone()
    assert row[0] == "done", f"expected status=done, got {row[0]}"
    con.close()


def test_short_run_with_result_completes(tmp_hermes_home):
    """Worker exits rc=0 with a result, even if <30s → marked done."""
    hermes, db_path = tmp_hermes_home
    from hermes_cli import kanban_db
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    now = int(time.time())
    _insert_task(con)
    _insert_run(con, "t_test", started_at=now - 5, ended_at=now)
    con.close()

    result = kanban_db.complete_task(
        task_id="t_test",
        summary=None,
        result="DID_SOMETHING_REAL",
    )
    assert result is True

    con = sqlite3.connect(db_path)
    row = con.execute("SELECT status FROM tasks WHERE id='t_test'").fetchone()
    assert row[0] == "done", f"expected status=done, got {row[0]}"
    con.close()


def test_env_var_override_disables_gate(tmp_hermes_home, monkeypatch):
    """HERMES_KANBAN_MIN_WORKER_RUNTIME_SECONDS=0 disables the gate."""
    monkeypatch.setenv("HERMES_KANBAN_MIN_WORKER_RUNTIME_SECONDS", "0")
    hermes, db_path = tmp_hermes_home
    from hermes_cli import kanban_db
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    now = int(time.time())
    _insert_task(con)
    _insert_run(con, "t_test", started_at=now - 5, ended_at=now)
    con.close()

    result = kanban_db.complete_task(
        task_id="t_test",
        summary=None,
        result=None,
    )
    assert result is True, "Expected gate disabled → completion succeeds"


def test_no_run_row_falls_through(tmp_hermes_home):
    """Manual completion without a run row → falls through, no crash."""
    hermes, db_path = tmp_hermes_home
    from hermes_cli import kanban_db
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    _insert_task(con)
    # No run row inserted
    con.close()

    # Should not crash; falls through to standard completion
    result = kanban_db.complete_task(
        task_id="t_test",
        summary="manual review",
        result=None,
    )
    assert result is True


def test_does_not_use_runs_table(tmp_hermes_home):
    """Regression: the fix must query task_runs, never the (non-existent) runs table."""
    hermes, db_path = tmp_hermes_home
    from hermes_cli import kanban_db
    # The buggy code path used `FROM runs WHERE task_id = ?` — a table that
    # doesn't exist. Verify the fix uses task_runs by checking the source.
    src = (REPO / "hermes_cli" / "kanban_db.py").read_text()
    assert "FROM runs WHERE" not in src, "buggy `FROM runs` is still in source"
    assert "FROM task_runs WHERE" in src, "fixed `FROM task_runs` not found"
