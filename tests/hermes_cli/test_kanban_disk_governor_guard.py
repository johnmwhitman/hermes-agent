"""Disk-governor admission for autonomous Kanban dispatch.

The portfolio disk governor already stops discretionary cron launches.  The
Kanban dispatcher must honor the same explicit state before it claims a ready
task; otherwise creating a card can bypass the resource gate and spawn a
worker immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import config as config_mod
from hermes_cli import config_defaults
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_system_memory_sample", lambda: {})
    kb.init_db()
    return home


def _write_state(path: Path, level: str) -> None:
    path.write_text(
        json.dumps(
            {
                "inventory": {
                    "schema": "disk-swap-governor.v1",
                    "pressure": {"level": level},
                    "read_only": level == "RED",
                },
                "plan": {
                    "pressure_level": level,
                    "read_only": level == "RED",
                    "automatic_execution_allowed": level != "RED",
                },
            }
        ),
        encoding="utf-8",
    )


def _configure_path(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(
        kb, "_configured_dispatch_governor_state_path", lambda: path
    )


def test_dispatch_governor_path_is_opt_in(monkeypatch, tmp_path):
    assert config_defaults.DEFAULT_CONFIG["kanban"]["disk_governor_state_path"] is None
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})
    assert kb._configured_dispatch_governor_state_path() is None

    expected = tmp_path / "state.json"
    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {
            "kanban": {"disk_governor_state_path": str(expected)}
        },
    )
    assert kb._configured_dispatch_governor_state_path() == expected


@pytest.mark.parametrize(
    "config",
    [
        {"kanban": []},
        {"kanban": {"disk_governor_state_path": ""}},
        {"kanban": {"disk_governor_state_path": "   "}},
        {"kanban": {"disk_governor_state_path": 123}},
    ],
)
def test_dispatch_governor_invalid_config_fails_closed(monkeypatch, config):
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)
    assert kb._dispatch_governor_level() == "UNKNOWN"


@pytest.mark.parametrize("level", ["GREEN", "YELLOW", "RED"])
def test_dispatch_governor_reads_known_level(monkeypatch, tmp_path, level):
    state = tmp_path / "state.json"
    _write_state(state, level)
    _configure_path(monkeypatch, state)
    assert kb._dispatch_governor_level() == level


@pytest.mark.parametrize("payload", ["", "{", '{"inventory": {}}'])
def test_dispatch_governor_fails_closed_on_missing_or_invalid_state(
    monkeypatch, tmp_path, payload
):
    state = tmp_path / "state.json"
    if payload:
        state.write_text(payload, encoding="utf-8")
    _configure_path(monkeypatch, state)
    assert kb._dispatch_governor_level() == "UNKNOWN"


def test_dispatch_governor_rejects_oversized_state(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    state.write_bytes(b" " * (kb.DISPATCH_GOVERNOR_MAX_BYTES + 1))
    _configure_path(monkeypatch, state)
    assert kb._dispatch_governor_level() == "UNKNOWN"


def test_dispatch_red_defers_ready_task_without_claiming(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="deferred", assignee="alice")
        events_before = len(kb.list_events(conn, task_id))
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: spawns.append(task.id),
        )
        task = kb.get_task(conn, task_id)
        run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        events_after = kb.list_events(conn, task_id)

    assert spawns == []
    assert result.spawned == []
    assert result.disk_pressure == "RED"
    assert task is not None and task.status == "ready"
    assert task.claim_lock is None
    assert run_count == 0
    assert len(events_after) == events_before
    assert all(event.kind != "claimed" for event in events_after)


def test_dispatch_unknown_governor_state_fails_closed(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    state = tmp_path / "missing.json"
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="deferred", assignee="alice")
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: spawns.append(task.id),
        )
        task = kb.get_task(conn, task_id)

    assert spawns == []
    assert result.disk_pressure == "UNKNOWN"
    assert task is not None and task.status == "ready"


@pytest.mark.parametrize("level", ["GREEN", "YELLOW"])
def test_dispatch_non_red_governor_state_allows_spawn(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path, level
):
    state = tmp_path / "state.json"
    _write_state(state, level)
    _configure_path(monkeypatch, state)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="allowed", assignee="alice")
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [task_id]
    assert result.disk_pressure is None


def test_dispatch_red_preserves_running_work_and_recomputes_ready(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)

    with kb.connect() as conn:
        running_id = kb.create_task(conn, title="running", assignee="alice")
        running_before = kb.claim_task(conn, running_id, claimer="worker:1")
        assert running_before is not None
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(
            conn, title="child", assignee="alice", parents=[parent]
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))

        # Saturate the concurrency cap too: disk pressure must remain the
        # observable admission reason instead of disappearing behind an
        # earlier cap short-circuit.
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: 42,
            max_in_progress=1,
        )
        running_after = kb.get_task(conn, running_id)
        child_after = kb.get_task(conn, child)

    assert result.disk_pressure == "RED"
    assert running_after is not None and running_after.status == "running"
    assert running_after.current_run_id == running_before.current_run_id
    assert child_after is not None and child_after.status == "ready"


def test_same_ready_task_dispatches_after_red_changes_to_green(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="defer then run", assignee="alice")
        blocked = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        _write_state(state, "GREEN")
        allowed = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, task_id)

    assert blocked.disk_pressure == "RED"
    assert blocked.spawned == []
    assert allowed.disk_pressure is None
    assert spawns == [task_id]
    assert task is not None and task.status == "running"


def test_dispatch_red_defers_review_lane_without_new_run_or_claim_event(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review deferred", assignee="alice")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        runs_before = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        events_before = len(kb.list_events(conn, task_id))

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: spawns.append(task.id),
        )

        task = kb.get_task(conn, task_id)
        runs_after = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        events_after = kb.list_events(conn, task_id)

    assert result.disk_pressure == "RED"
    assert spawns == []
    assert task is not None and task.status == "review"
    assert task.claim_lock is None
    assert runs_after == runs_before
    assert len(events_after) == events_before
    assert all(event.kind != "claimed" for event in events_after)
