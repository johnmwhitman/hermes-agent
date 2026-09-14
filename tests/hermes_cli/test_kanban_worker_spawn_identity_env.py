"""Regression: naturally dispatched workers receive task/run/branch/workspace identity.

Follow-up to native activation canary t_31d82437. The dispatcher ``_default_spawn``
already pinned ``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_RUN_ID`` / ``HERMES_KANBAN_BRANCH``
and ``TERMINAL_CWD`` to the task workspace (see #34619 / #41312), but the parent task
documented a separate failure: a worker spawned by the dispatcher still ended up with
``HERMES_KANBAN_TASK=`` and ``HERMES_KANBAN_RUN_ID=`` empty when the dispatching process
was itself a delegate_task child of the gateway. This test suite pins the contract for
all three workspace kinds (scratch/dir/worktree) and across the dirty-context scenarios
the parent canary surfaced:

1. The dispatcher is itself a delegated child (gateway's ``HERMES_DELEGATED_CHILD_CONTEXT=1``
   is in the dispatcher's env). ``scrub_kanban_env`` must NOT strip the new worker's
   freshly-set identity vars back out of the spawned subprocess env.
2. The parent's own ``HERMES_KANBAN_TASK`` (different id) is set — the new worker must
   receive the *claimed* task's id, not the parent's.
3. ``cwd`` argument and ``HERMES_KANBAN_WORKSPACE`` agree with ``TERMINAL_CWD``.
4. ``HERMES_KANBAN_RUN_ID`` carries the *claimed* run id and is parsed as int.
5. ``HERMES_KANBAN_BRANCH`` is set for worktree tasks, unset for scratch/dir.
6. ``HERMES_KANBAN_CLAIM_LOCK`` carries the dispatcher's claim lock so the worker can
   heartbeat/close its own claim without races.
7. ``HERMES_DELEGATED_CHILD_CONTEXT`` is consumed by the dispatcher (the new worker is
   itself a delegate_task child of the *worker*, not of the gateway's chrome).

Tests do NOT spawn real Hermes subprocesses; ``subprocess.Popen`` is intercepted so the
assertions can target the env contract directly. The tests are intentionally narrowly
scoped so they keep running when the wider spawn pipeline (managed systemd, restart-safe
scope, board routing) is refactored.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


def _capture_popen(monkeypatch) -> list[dict]:
    """Intercept subprocess.Popen and return the captured (cmd, env, cwd) tuples."""
    captured: list[dict] = []

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured.append({
            "cmd": list(cmd),
            "env": dict(kwargs.get("env") or {}),
            "cwd": kwargs.get("cwd"),
        })
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return captured


def _make_task(*, kind: str, run_id: int = 12345, branch: str | None = None,
               workspace_path: str | None = None) -> kb.Task:
    return kb.Task(
        id=f"t_{kind}_worker",
        title=f"{kind} worker identity",
        body=None,
        assignee="w",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind=kind,
        workspace_path=workspace_path,
        claim_lock="Mac.localdomain:27857",
        claim_expires=None,
        tenant=None,
        current_run_id=run_id,
        branch_name=branch,
    )


def _setup_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    return root


# ---------------------------------------------------------------------------
# Cross-kind contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["scratch", "dir", "worktree"])
def test_default_spawn_propagates_task_and_run_id(monkeypatch, tmp_path, kind):
    """HERMES_KANBAN_TASK and HERMES_KANBAN_RUN_ID survive _default_spawn for every kind."""
    _setup_hermes_home(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    branch = "wt/branch" if kind == "worktree" else None
    task = _make_task(kind=kind, run_id=12345, branch=branch,
                      workspace_path=str(workspace) if kind != "scratch" else None)
    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(task, str(workspace))
    assert captured, "Popen was not intercepted"
    env = captured[0]["env"]
    assert env["HERMES_KANBAN_TASK"] == task.id
    assert env["HERMES_KANBAN_RUN_ID"] == "12345"
    assert int(env["HERMES_KANBAN_RUN_ID"]) == task.current_run_id


@pytest.mark.parametrize("kind", ["scratch", "dir", "worktree"])
def test_default_spawn_cwd_matches_workspace(monkeypatch, tmp_path, kind):
    """Subprocess cwd, TERMINAL_CWD, and HERMES_KANBAN_WORKSPACE all point at the workspace."""
    _setup_hermes_home(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    branch = "wt/branch" if kind == "worktree" else None
    task = _make_task(kind=kind, branch=branch,
                      workspace_path=str(workspace) if kind != "scratch" else None)
    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(task, str(workspace))
    env = captured[0]["env"]
    assert captured[0]["cwd"] == str(workspace)
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["TERMINAL_CWD"] == str(workspace)


def test_default_spawn_sets_branch_only_for_worktree(monkeypatch, tmp_path):
    """HERMES_KANBAN_BRANCH is set for worktree tasks, omitted otherwise."""
    _setup_hermes_home(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(
        kb.Task(
            id="t_dir_branch", title="dir", body=None, assignee="w",
            status="running", priority=0, created_by="t", created_at=1,
            started_at=None, completed_at=None,
            workspace_kind="dir", workspace_path=str(workspace),
            claim_lock="l", claim_expires=None, tenant=None,
            current_run_id=1, branch_name=None,
        ),
        str(workspace),
    )
    assert "HERMES_KANBAN_BRANCH" not in captured[0]["env"]

    captured.append({"env": {}, "cwd": None})  # marker so the next append is the worktree
    kbd._default_spawn(
        kb.Task(
            id="t_wt_branch", title="wt", body=None, assignee="w",
            status="running", priority=0, created_by="t", created_at=1,
            started_at=None, completed_at=None,
            workspace_kind="worktree", workspace_path=str(workspace),
            claim_lock="l", claim_expires=None, tenant=None,
            current_run_id=1, branch_name="wt/wt",
        ),
        str(workspace),
    )
    assert captured[-1]["env"]["HERMES_KANBAN_BRANCH"] == "wt/wt"


def test_default_spawn_propagates_claim_lock(monkeypatch, tmp_path):
    """Claim lock rides into the worker env so heartbeats/close can race-free address the row."""
    _setup_hermes_home(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _make_task(kind="dir", workspace_path=str(workspace))
    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(task, str(workspace))
    assert captured[0]["env"]["HERMES_KANBAN_CLAIM_LOCK"] == task.claim_lock


# ---------------------------------------------------------------------------
# Dirty-context guards (parent task canary surface)
# ---------------------------------------------------------------------------

def test_spawn_isolates_task_id_from_parent(monkeypatch, tmp_path):
    """When the dispatching process already has HERMES_KANBAN_TASK set to a parent task,
    the new worker must receive the *claimed* task id, not the inherited parent id."""
    _setup_hermes_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent_session")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "11111")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "Mac.localdomain:00000")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _make_task(kind="dir", workspace_path=str(workspace), run_id=22222)
    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(task, str(workspace))
    env = captured[0]["env"]
    assert env["HERMES_KANBAN_TASK"] == task.id
    assert env["HERMES_KANBAN_TASK"] != "t_parent_session"
    assert env["HERMES_KANBAN_RUN_ID"] == "22222"
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == task.claim_lock


def test_spawn_consumes_delegated_child_marker(monkeypatch, tmp_path):
    """A dispatcher running inside a delegate_task child must not forward
    HERMES_DELEGATED_CHILD_CONTEXT to its spawned worker — the worker is the
    new top-level dispatcher-owned context, not a delegate child of the gateway."""
    _setup_hermes_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _make_task(kind="dir", workspace_path=str(workspace))
    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(task, str(workspace))
    assert "HERMES_DELEGATED_CHILD_CONTEXT" not in captured[0]["env"]


def test_spawn_rejects_empty_workspace(monkeypatch, tmp_path):
    """An empty/non-existent workspace must NOT silently fall back to os.getcwd() — the
    worker would inherit the gateway's cwd. The dispatcher must surface the absence."""
    _setup_hermes_home(tmp_path, monkeypatch)
    task = _make_task(kind="dir", workspace_path=str(tmp_path / "missing"))
    captured = _capture_popen(monkeypatch)
    # ``workspace=`` is the literal string the dispatcher was handed; if it doesn't
    # resolve to a real directory, the dispatcher must NOT silently fall back to the
    # gateway's process cwd (the live defect documented in the parent canary).
    kbd._default_spawn(task, "")
    assert captured, "Popen was not intercepted"
    env = captured[0]["env"]
    cwd = captured[0]["cwd"]
    # When workspace is empty, neither cwd nor TERMINAL_CWD may pin to the gateway cwd.
    assert cwd in (None, ""), (
        f"empty workspace must not become cwd={cwd!r}; "
        "this is the parent canary failure shape"
    )
    # And no fake HERMES_KANBAN_WORKSPACE leak.
    if "HERMES_KANBAN_WORKSPACE" in env:
        assert env["HERMES_KANBAN_WORKSPACE"] != os.getcwd(), (
            f"empty workspace must not become HERMES_KANBAN_WORKSPACE={env['HERMES_KANBAN_WORKSPACE']!r}"
        )


def test_spawn_strips_session_routing_leak(monkeypatch, tmp_path):
    """Inherited HERMES_SESSION_* routing must not leak into a worker subprocess."""
    _setup_hermes_home(tmp_path, monkeypatch)
    # Simulate a long-lived gateway that routed a previous turn.
    for key in (
        "HERMES_SESSION_ID",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_OWNER_HANDLE",
    ):
        monkeypatch.setenv(key, "stale-routing-value")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = _make_task(kind="dir", workspace_path=str(workspace))
    captured = _capture_popen(monkeypatch)
    kbd._default_spawn(task, str(workspace))
    env = captured[0]["env"]
    for key in (
        "HERMES_SESSION_ID",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_OWNER_HANDLE",
    ):
        assert key not in env, (
            f"{key} leaked into worker env: {env.get(key)!r}"
        )
    # HERMES_SESSION_SOURCE is the one exception — the dispatcher tags it "kanban".
    assert env.get("HERMES_SESSION_SOURCE") == "kanban"
