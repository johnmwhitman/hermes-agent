"""Regression tests for t_cca67626 — receipt SHA resolver pins to /Users/johnwhitman/AI.

The card body documents a Hermes framework P1 bug: the receipt
classifier used to hard-code its SHA resolver against
``/Users/johnwhitman/AI`` regardless of which repo the card actually
committed to. The bug dropped 106 completions' receipt_outbox rows
because their git_commit SHAs did not resolve in the umbrella repo.

These tests pin the new contract end-to-end at the ``build_envelope``
layer:

* A scratch-workspace card whose commit lives outside ``~/AI`` resolves
  its SHA against its own worktree and emits a ``git_commit`` evidence
  entry. (Acceptance #1 from the card body.)
* Replaying the close path inserts a pending ``kanban_receipt_outbox``
  row — the durable receipt signal that was silently dropped before.
  (Acceptance #2.)
* The classifier's backwards-compat default (``~/AI`` only) is preserved
  for callers that pre-date the multi-repo fix.

The literal t_ae9975c9 SHA is used as the byte case where feasible so a
production-failure-class fix lands a test that proves it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_git_repo(path, with_commit: bool = True) -> str | None:
    """Initialize a git repo at ``path`` and optionally commit a file.

    Returns the HEAD SHA when ``with_commit=True``, otherwise None.
    Configures a local identity so the commit does not fail on CI hosts
    that lack a global user.email/name.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "--initial-branch", "main"],
        cwd=str(path), check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "t_cca67626_test"],
        check=True,
    )
    if not with_commit:
        return None
    (path / "marker.txt").write_text("t_cca67626 fixture")
    subprocess.run(["git", "-C", str(path), "add", "marker.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()


def _seed_minimal_kanban_db(
    tmp_path,
    *,
    task_id: str,
    workspace_path: str | None,
    project_id: str | None = None,
    result_text: str,
    assignee: str = "platformops",
    workspace_kind: str = "scratch",
    branch_name: str | None = None,
) -> sqlite3.Connection:
    """Create a throwaway kanban.db with the schema shape ``build_envelope``
    and ``_task_repo_candidates`` read from.

    Returns the open connection with row_factory=sqlite3.Row.
    """
    schema = """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        title TEXT,
        body TEXT,
        assignee TEXT,
        status TEXT,
        workspace_kind TEXT DEFAULT 'scratch',
        workspace_path TEXT,
        project_id TEXT,
        branch_name TEXT,
        result TEXT,
        created_at INTEGER,
        started_at INTEGER,
        completed_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS task_attachments (
        task_id TEXT,
        stored_path TEXT,
        sha256 TEXT
    );
    """
    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    for stmt in schema.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt)
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, workspace_kind, "
        "workspace_path, project_id, branch_name, result, created_at, "
        "started_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            task_id, "fixture", "fixture body", assignee, "done",
            workspace_kind, workspace_path, project_id, branch_name,
            result_text, 1, 1, int(time.time()),
        ),
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# build_envelope byte case — pinned to a non-~/AI SHA (t_cca67626)
# ---------------------------------------------------------------------------


def test_build_envelope_emits_git_commit_for_external_repo_sha(tmp_path):
    """A card whose commit lives outside ``~/AI`` produces evidence with
    a ``git_commit`` handle for that SHA and does NOT raise
    ResultContractError. Mirrors the literal t_ae9975c9 failure mode."""
    from hermes_cli.kanban_receipt import (
        build_envelope,
        ResultContractError,
        validate_envelope,
    )

    repo = tmp_path / "external_repo"
    sha = _make_git_repo(repo)
    assert sha is not None and len(sha) >= 10

    # Result text shaped like t_ae9975c9's stored body.
    result_text = (
        f"VERIFIED: t_cca67626_test | external SHA in non-~/AI repo | "
        f"commit {sha} pushed to local | "
        f"receipt handles: {repo}/marker.txt | "
        f"command evidence: `python -m pytest -q` => ok"
    )
    conn = _seed_minimal_kanban_db(
        tmp_path,
        task_id="t_extrepo01",
        workspace_path=str(repo),
        result_text=result_text,
    )

    envelope = build_envelope(
        conn, "t_extrepo01",
        run_id=1,
        assignee="platformops",
        completed_at=int(time.time()),
        summary=result_text.splitlines()[0],
        result=result_text,
        metadata={},
        artifacts=[],
        task_status="done",
    )

    # Acceptance #1: evidence contains a git_commit handle for the SHA.
    handles = [e["handle"] for e in envelope["evidence"] if e["kind"] == "git_commit"]
    assert sha in handles, envelope
    # Result contract agrees the work was good.
    assert envelope["quality_gate"] == "passed", envelope
    assert envelope["result_contract"] == "ok", envelope
    assert envelope["terminal_outcome"] == "completed", envelope

    # Wire-side validate_envelope also passes (no drift).
    reasons = validate_envelope(envelope)
    assert reasons == [], reasons

    conn.close()


def test_build_envelope_outbox_insert_succeeds_for_external_repo_sha(tmp_path):
    """Acceptance #2: replaying the close path that drops the outbox
    row succeeds for a card whose commit lives outside ``~/AI``.

    Mirrors the ``INSERT INTO kanban_receipt_outbox ... ON CONFLICT
    DO NOTHING`` pattern from ``kanban_db._enqueue_receipt_outbox``
    verbatim. The test asserts the row count delta is exactly +1.
    """
    from hermes_cli.kanban_receipt import (
        build_envelope,
        compute_payload_sha256,
    )

    repo = tmp_path / "external_repo"
    sha = _make_git_repo(repo)
    assert sha is not None

    result_text = (
        f"VERIFIED: outbox round-trip for t_cca67626 | "
        f"commit {sha} | receipt handles: {repo}/marker.txt"
    )
    conn = _seed_minimal_kanban_db(
        tmp_path,
        task_id="t_extrepo02",
        workspace_path=str(repo),
        result_text=result_text,
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kanban_receipt_outbox ("
        "  task_id TEXT, run_id INTEGER, payload_sha256 TEXT, "
        "  payload_json TEXT, created_at INTEGER, status TEXT DEFAULT 'pending',"
        "  UNIQUE(task_id, run_id))"
    )
    conn.commit()

    envelope = build_envelope(
        conn, "t_extrepo02",
        run_id=42,
        assignee="platformops",
        completed_at=int(time.time()),
        summary=result_text.splitlines()[0],
        result=result_text,
        metadata={},
        artifacts=[],
        task_status="done",
    )

    digest = compute_payload_sha256(envelope)
    payload_json = json.dumps(envelope, ensure_ascii=False, sort_keys=True)
    before = conn.execute(
        "SELECT COUNT(*) c FROM kanban_receipt_outbox WHERE task_id=? AND run_id=?",
        ("t_extrepo02", 42),
    ).fetchone()["c"]
    conn.execute(
        "INSERT INTO kanban_receipt_outbox (task_id, run_id, payload_sha256, "
        "payload_json, created_at, status) VALUES (?,?,?,?,?, 'pending') "
        "ON CONFLICT(task_id, run_id) DO NOTHING",
        ("t_extrepo02", 42, digest, payload_json, int(time.time())),
    )
    conn.commit()
    after = conn.execute(
        "SELECT COUNT(*) c FROM kanban_receipt_outbox WHERE task_id=? AND run_id=?",
        ("t_extrepo02", 42),
    ).fetchone()["c"]

    assert after == before + 1, f"expected delta=1, got {after - before}"

    # Idempotency: replaying the same insert leaves the count unchanged.
    conn.execute(
        "INSERT INTO kanban_receipt_outbox (task_id, run_id, payload_sha256, "
        "payload_json, created_at, status) VALUES (?,?,?,?,?, 'pending') "
        "ON CONFLICT(task_id, run_id) DO NOTHING",
        ("t_extrepo02", 42, digest, payload_json, int(time.time())),
    )
    conn.commit()
    after2 = conn.execute(
        "SELECT COUNT(*) c FROM kanban_receipt_outbox WHERE task_id=? AND run_id=?",
        ("t_extrepo02", 42),
    ).fetchone()["c"]
    assert after2 == after, "idempotency violated on replay"

    conn.close()


def test_build_envelope_reproduction_byte_case_t_ae9975c9(tmp_path):
    """Reproduce the literal t_ae9975c9 byte case against the live SHA.

    Skips gracefully when the commit is not reachable in the test host's
    hermes-agent checkout (CI may run against a different tree). On a
    normal dev box this exercises the production-failure SHA directly.
    """
    sha = "5f71160c9b5235ff8fa1c744085a31090dc88ad1"
    agent_repo = "/Users/johnwhitman/.config/hermes-lane/data/hermes-agent"
    if not os.path.exists(agent_repo):
        pytest.skip("agent repo not on this host")
    try:
        ok = subprocess.run(
            ["git", "-C", agent_repo, "cat-file", "-e", sha + "^{commit}"],
            check=False, timeout=10,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        pytest.skip(f"sha {sha} not present in {agent_repo}")

    from hermes_cli.kanban_receipt import build_envelope

    worktree = f"{agent_repo}/.worktrees/t_ae9975c9"
    result_text = (
        f"VERIFIED: t_ae9975c9 | heartbeat watch registry now survives "
        f"Desktop orphan respawn | commit {sha} on wt/t_ae9975c9 pushed to "
        f"private-offsite | receipt handles: {worktree}/hermes_cli/heartbeat.py"
    )
    conn = _seed_minimal_kanban_db(
        tmp_path,
        task_id="t_ae9975c9",
        workspace_path=worktree,
        result_text=result_text,
    )

    envelope = build_envelope(
        conn, "t_ae9975c9",
        run_id=13403,
        assignee="platformops",
        completed_at=int(time.time()),
        summary=result_text.splitlines()[0],
        result=result_text,
        metadata={},
        artifacts=[],
        task_status="done",
    )
    handles = [e["handle"] for e in envelope["evidence"] if e["kind"] == "git_commit"]
    assert sha in handles, envelope
    conn.close()


# ---------------------------------------------------------------------------
# Backwards-compat — no candidate list ⇒ ~/AI single-root default
# ---------------------------------------------------------------------------


def test_classify_prose_uses_task_workspace_when_no_reps_passed(tmp_path):
    """``_classify_prose`` derives the candidate list from the task row
    when the caller does not pass ``repos``. This is the path
    ``build_envelope`` exercises."""
    from hermes_cli.kanban_receipt import _classify_prose

    repo = tmp_path / "ws_repo"
    sha = _make_git_repo(repo)
    assert sha is not None

    # Direct call without ``repos`` ⇒ the helper must look up
    # workspace_path from the task row and resolve against that repo.
    conn = _seed_minimal_kanban_db(
        tmp_path,
        task_id="t_classify01",
        workspace_path=str(repo),
        result_text=f"VERIFIED: commit {sha}",
    )
    prose = _classify_prose(
        conn, "t_classify01",
        summary="VERIFIED",
        result=f"VERIFIED: commit {sha}",
        artifacts=[],
    )
    assert sha in prose["existing_shas"], prose
    conn.close()


def test_classify_prose_falls_back_to_ai_root_for_missing_workspace(tmp_path):
    """When ``workspace_path`` is NULL, the candidate list is just the
    historical ``~/AI`` fallback. SHAs that resolve there come through;
    SHAs that don't are correctly absent from ``existing_shas``."""
    from hermes_cli.kanban_receipt import _classify_prose

    conn = _seed_minimal_kanban_db(
        tmp_path,
        task_id="t_classify02",
        workspace_path=None,
        result_text="VERIFIED: nothing resolves here",
    )
    prose = _classify_prose(
        conn, "t_classify02",
        summary="",
        result="VERIFIED: nothing resolves here",
        artifacts=[],
    )
    assert prose["existing_shas"] == []
    assert prose["has_verified"] is True
    conn.close()
