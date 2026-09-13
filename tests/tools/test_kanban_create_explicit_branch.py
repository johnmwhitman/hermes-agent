"""kanban_create must persist optional branch_name (CLI --branch parity).

The CLI already stores create --branch on worktree tasks. The agent-facing
kanban_create tool used to drop that field, so an explicit existing recovery
branch was lost and the claim resolver fell back to wt/<id>. These tests
drive the tool handler/schema plus the existing resolver — never the live
board. Temp HERMES_HOME / git repos are established before handler imports.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


RECOVERY_BRANCH = "recovery/occupied"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME + empty kanban DB before any tool-handler import."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture
def worker_env(isolated_home, monkeypatch):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def _head_branch(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _create(args: dict) -> dict:
    from tools import kanban_tools as kt
    return json.loads(kt._handle_create(args))


def test_kanban_create_schema_exposes_branch_name():
    """Agent-facing schema must name the DB field and document CLI --branch."""
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA

    props = KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert "branch_name" in props
    assert props["branch_name"]["type"] == "string"
    desc = props["branch_name"]["description"]
    assert "branch_name" in desc
    assert "--branch" in desc


def test_explicit_recovery_branch_stays_in_registered_worktree(
    worker_env, tmp_path,
):
    """Passing an existing recovery branch must store it and keep the checkout."""
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(
        repo, repo / ".worktrees" / "recovery-occupied", RECOVERY_BRANCH,
    )

    out = _create({
        "title": "resume recovery",
        "assignee": "peer",
        "workspace_kind": "worktree",
        "workspace_path": str(occupied),
        "branch_name": RECOVERY_BRANCH,
    })
    assert out.get("ok") is True, out
    assert out["branch_name"] == RECOVERY_BRANCH
    assert out["workspace_kind"] == "worktree"
    assert out["workspace_path"] == str(occupied)

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, out["task_id"])
    finally:
        conn.close()
    assert task.branch_name == RECOVERY_BRANCH
    assert task.workspace_path == str(occupied)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == occupied.resolve()
    assert branch == RECOVERY_BRANCH
    assert _head_branch(occupied) == RECOVERY_BRANCH


def test_omitted_branch_on_occupied_path_falls_back(worker_env, tmp_path):
    """Omitted branch_name keeps the protective fallback; occupied checkout stays."""
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(
        repo, repo / ".worktrees" / "recovery-occupied", RECOVERY_BRANCH,
    )

    out = _create({
        "title": "omitted branch",
        "assignee": "peer",
        "workspace_kind": "worktree",
        "workspace_path": str(occupied),
    })
    assert out.get("ok") is True, out
    assert out.get("branch_name") in (None, "")

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, out["task_id"])
    finally:
        conn.close()
    assert task.branch_name is None

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / task.id).resolve()
    assert branch == f"wt/{task.id}"
    assert _head_branch(occupied) == RECOVERY_BRANCH


def test_different_branch_on_occupied_path_falls_back(worker_env, tmp_path):
    """A differently named requested branch must not reuse the occupied checkout."""
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(
        repo, repo / ".worktrees" / "recovery-occupied", RECOVERY_BRANCH,
    )
    requested = "wt/other-task"

    out = _create({
        "title": "different branch",
        "assignee": "peer",
        "workspace_kind": "worktree",
        "workspace_path": str(occupied),
        "branch_name": requested,
    })
    assert out.get("ok") is True, out
    assert out["branch_name"] == requested

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, out["task_id"])
    finally:
        conn.close()
    assert task.branch_name == requested

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace != occupied.resolve()
    assert workspace == (repo / ".worktrees" / task.id).resolve()
    assert branch == requested
    assert _head_branch(occupied) == RECOVERY_BRANCH


def test_scratch_default_omits_branch(worker_env):
    """Ordinary omitted-branch scratch create stays scratch with no branch."""
    out = _create({"title": "plain child", "assignee": "peer"})
    assert out.get("ok") is True, out
    assert out["workspace_kind"] == "scratch"
    assert out.get("branch_name") in (None, "")

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, out["task_id"])
    finally:
        conn.close()
    assert task.workspace_kind == "scratch"
    assert task.branch_name is None


def test_branch_name_rejected_without_worktree(worker_env):
    """DB contract: branch_name is only valid for worktree workspaces."""
    out = _create({
        "title": "scratch with branch",
        "assignee": "peer",
        "workspace_kind": "scratch",
        "branch_name": RECOVERY_BRANCH,
    })
    assert "error" in out
    assert "branch_name" in out["error"]


def test_delegated_child_cannot_create(worker_env, monkeypatch):
    """Existing delegated-child mutation rejection is unchanged."""
    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools as kt

    with delegated_child_context():
        out = json.loads(kt._handle_create({
            "title": "child must not create",
            "assignee": "peer",
        }))
    assert "error" in out
    assert "delegate_task child" in out["error"]
