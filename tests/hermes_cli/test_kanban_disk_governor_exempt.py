"""Disk-governor RED exemption for triage profiles.

The disk governor stops discretionary dispatch when the host is under RED
disk/swap pressure. But conductor and overwatch are the lanes allowed to
issue the disk-pressure envelope and reclaim space — if the gate strands
them, nobody can ever relieve the pressure (measured 2026-09-04: the gate
blocked its own issuer for 82 minutes while five ready cards, a P1 backup
run among them, waited). These tests pin the exemption: exempt triage
profiles still spawn under RED; everyone else stays deferred.
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


def _spawn_recorder(spawns):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    return fake_spawn


def test_disk_governor_red_exempts_triage_profile(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    """RED must not strand conductor/overwatch — the gate's own issuer."""
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        conductor_id = kb.create_task(
            conn, title="triage under pressure", assignee="conductor"
        )
        overwatch_id = kb.create_task(
            conn, title="issue disk-pressure envelope", assignee="overwatch"
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns))

    assert result.disk_pressure == "RED"
    assert sorted(spawns) == sorted([conductor_id, overwatch_id])


def test_disk_governor_red_still_defers_non_exempt_profile(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    """Negative: a non-exempt profile is still deferred under RED."""
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="deferred", assignee="alice")
        events_before = len(kb.list_events(conn, task_id))
        result = kb.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns))
        task = kb.get_task(conn, task_id)
        run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        events_after = kb.list_events(conn, task_id)

    assert result.disk_pressure == "RED"
    assert result.spawned == []
    assert spawns == []
    assert task is not None and task.status == "ready"
    assert task.claim_lock is None
    assert run_count == 0
    assert len(events_after) == events_before


def test_disk_governor_red_mixed_board_spawns_only_exempt(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    """A mixed board under RED spawns only the exempt assignees."""
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        deferred_id = kb.create_task(conn, title="deferred", assignee="alice")
        exempt_id = kb.create_task(
            conn, title="envelope issuer", assignee="overwatch"
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns))
        deferred = kb.get_task(conn, deferred_id)
        exempt = kb.get_task(conn, exempt_id)

    assert result.disk_pressure == "RED"
    assert spawns == [exempt_id]
    assert deferred is not None and deferred.status == "ready"
    assert exempt is not None and exempt.status == "running"


def test_disk_governor_red_exempts_review_lane_too(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    """Exemption covers the review loop as well as the ready loop."""
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    spawns = []

    with kb.connect() as conn:
        review_id = kb.create_task(
            conn, title="review under pressure", assignee="conductor"
        )
        conn.execute(
            "UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,)
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns))

    assert result.disk_pressure == "RED"
    assert spawns == [review_id]


def test_disk_governor_exempt_profiles_default_matches_triage_set(
    monkeypatch,
):
    """The default exemption is the conductor triage set."""
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})
    assert kb._disk_governor_exempt_profiles() == frozenset(
        {"conductor", "overwatch"}
    )
    assert (
        config_defaults.DEFAULT_CONFIG["kanban"]["disk_governor_exempt_profiles"]
        is None
    )


def test_disk_governor_exempt_profiles_config_override(monkeypatch):
    """Operators can widen or narrow the exempt set via config."""
    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"kanban": {"disk_governor_exempt_profiles": ["ops", "overwatch"]}},
    )
    assert kb._disk_governor_exempt_profiles() == frozenset({"ops", "overwatch"})

    # Explicit empty list disables the exemption entirely.
    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"kanban": {"disk_governor_exempt_profiles": []}},
    )
    assert kb._disk_governor_exempt_profiles() == frozenset()


@pytest.mark.parametrize(
    "config",
    [
        {"kanban": []},
        {"kanban": {"disk_governor_exempt_profiles": "overwatch"}},
        {"kanban": {"disk_governor_exempt_profiles": 123}},
    ],
)
def test_disk_governor_exempt_profiles_invalid_config_fails_to_default(
    monkeypatch, config
):
    """Invalid config fails closed to the default triage set."""
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)
    assert kb._disk_governor_exempt_profiles() == frozenset(
        {"conductor", "overwatch"}
    )


def test_disk_governor_red_with_empty_exemption_defers_everything(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    """An explicitly empty exemption restores the old fail-closed behavior."""
    state = tmp_path / "state.json"
    _write_state(state, "RED")
    _configure_path(monkeypatch, state)
    monkeypatch.setattr(
        kb, "_disk_governor_exempt_profiles", lambda: frozenset()
    )
    spawns = []

    with kb.connect() as conn:
        kb.create_task(conn, title="triage under pressure", assignee="conductor")
        result = kb.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns))

    assert result.disk_pressure == "RED"
    assert result.spawned == []
    assert spawns == []


def test_disk_governor_unknown_state_also_exempts_triage(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path
):
    """UNKNOWN (missing/malformed configured state) exempts the same set."""
    _configure_path(monkeypatch, tmp_path / "missing.json")
    spawns = []

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="triage under unknown state", assignee="overwatch"
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns))

    assert result.disk_pressure == "UNKNOWN"
    assert spawns == [task_id]
