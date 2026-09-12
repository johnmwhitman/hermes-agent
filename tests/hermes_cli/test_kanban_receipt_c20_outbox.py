"""C20 regressions: receipt_outbox_rejected still firing after t_02d3dba9.

Live rejects in ~/AI/agents/.hermes/kanban.db (t_2531481b) were two classes:

1. ``ResultContractError('quality_gate=passed requires at least one evidence
   entry')`` — honest VERIFIED completions that cite an on-disk path (the
   classifier's ``existing_paths``) but ``_resolve_evidence`` emits nothing
   for those paths, so ``_derive_triple`` claims ``quality_gate=passed``
   and the cross-field check refuses the envelope. Live examples:
   t_416b98ce, t_27314bf6, t_41ac56b4.

2. ``ResultContractError('assignee must be a non-empty string, got None')``
   — system-created ``kanban-rtt-probe-*`` cards have a NULL assignee.
   ``_enqueue_receipt_outbox`` must coalesce that to ``\"system\"`` so the
   close still writes an outbox row instead of a reject event.

Do not mark C20 done from a refactor PR alone: these tests pin the two
byte cases that were still firing after t_02d3dba9 / t_cca67626 /
t_d7980ed0 closed.
"""
from __future__ import annotations

import json
import sqlite3
import time

from hermes_cli.kanban_receipt import (
    build_envelope,
    validate_envelope,
)


def _seed_db(
    tmp_path,
    *,
    task_id: str,
    result_text: str,
    assignee: str | None,
    workspace_path: str | None = None,
) -> sqlite3.Connection:
    """Minimal schema for ``build_envelope`` + ``_enqueue_receipt_outbox``."""
    conn = sqlite3.connect(str(tmp_path / "kanban.db"))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE tasks (
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
        CREATE TABLE task_attachments (
            task_id TEXT,
            stored_path TEXT,
            sha256 TEXT
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            run_id INTEGER,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE kanban_receipt_outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_attempt_at INTEGER,
            last_error TEXT,
            lease_expires_at INTEGER,
            UNIQUE(task_id, run_id)
        );
        """
    )
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, workspace_kind, "
        "workspace_path, project_id, branch_name, result, created_at, "
        "started_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            task_id,
            "fixture",
            "fixture body",
            assignee,
            "done",
            "scratch",
            workspace_path,
            None,
            None,
            result_text,
            1,
            1,
            int(time.time()),
        ),
    )
    conn.commit()
    return conn


def test_build_envelope_emits_artifact_for_existing_path_without_sha(tmp_path):
    """Live C20 class 1: VERIFIED + on-disk path, no git SHA, no attachment.

    Classifier admits the close (``existing_paths`` is non-empty). The
    envelope builder used to raise ResultContractError because it never
    serialized those paths as wire evidence while still stamping
    ``quality_gate=passed``. After the fix the envelope stores, does not
    reject, and carries at least one evidence entry.
    """
    receipt_file = tmp_path / "c20-receipt.txt"
    receipt_file.write_text("c20 existing_path fixture\n")

    result_text = (
        "VERIFIED via sqlite3 against "
        f"{receipt_file} : C20 path-only receipt self-healed. "
        "No git SHA in this summary on purpose."
    )
    conn = _seed_db(
        tmp_path,
        task_id="t_c20path01",
        result_text=result_text,
        assignee="conductor",
        workspace_path=str(tmp_path),
    )

    envelope = build_envelope(
        conn,
        "t_c20path01",
        run_id=1,
        assignee="conductor",
        completed_at=int(time.time()),
        summary=result_text,
        result=result_text,
        metadata={},
        artifacts=[],
        task_status="done",
    )

    assert envelope["quality_gate"] == "passed", envelope
    assert envelope["result_contract"] == "ok", envelope
    assert envelope["terminal_outcome"] == "completed", envelope
    assert len(envelope["evidence"]) >= 1, envelope
    kinds = {e["kind"] for e in envelope["evidence"]}
    assert "artifact" in kinds, envelope
    assert validate_envelope(envelope) == [], envelope
    conn.close()


def test_build_envelope_verified_cmd_without_wire_evidence_does_not_raise(tmp_path):
    """VERIFIED+cmd is a prose gate, not a wire evidence kind.

    The MeshFleet invariant forbids ``quality_gate=passed`` with an empty
    evidence array. The envelope must still *build* (fail-open card) rather
    than raise ``ResultContractError`` — otherwise every such close writes
    ``receipt_outbox_rejected`` and stores nothing.
    """
    result_text = (
        "VERIFIED: C20 prose-only command receipt with no path and no SHA.\n"
        "bash echo c20-prose-only"
    )
    conn = _seed_db(
        tmp_path,
        task_id="t_c20cmd01",
        result_text=result_text,
        assignee="platformops",
    )

    envelope = build_envelope(
        conn,
        "t_c20cmd01",
        run_id=2,
        assignee="platformops",
        completed_at=int(time.time()),
        summary=result_text,
        result=result_text,
        metadata={},
        artifacts=[],
        task_status="done",
    )

    assert envelope["evidence"] == [], envelope
    assert envelope["quality_gate"] == "failed", envelope
    assert envelope["result_contract"] == "absent", envelope
    assert validate_envelope(envelope) == [], envelope
    conn.close()


def test_build_envelope_coalesces_null_assignee(tmp_path):
    """C20 class 2 at the contract boundary: assignee=None must not raise."""
    receipt_file = tmp_path / "c20-assignee.txt"
    receipt_file.write_text("assignee coalesce fixture\n")
    result_text = f"VERIFIED: null-assignee envelope against {receipt_file}"
    conn = _seed_db(
        tmp_path,
        task_id="t_c20asg01",
        result_text=result_text,
        assignee=None,
        workspace_path=str(tmp_path),
    )
    envelope = build_envelope(
        conn,
        "t_c20asg01",
        run_id=4,
        assignee=None,  # type: ignore[arg-type]
        completed_at=int(time.time()),
        summary=result_text,
        result=result_text,
        metadata={},
        artifacts=[],
        task_status="done",
    )
    assert envelope["assignee"] == "system"
    assert validate_envelope(envelope) == []
    conn.close()


def test_enqueue_receipt_outbox_coalesces_null_assignee(tmp_path):
    """Live C20 class 2: kanban-rtt-probe cards have assignee NULL.

    ``_enqueue_receipt_outbox`` must not write ``receipt_outbox_rejected``
    with ``assignee must be a non-empty string, got None``. It coalesces
    to ``system`` and stores a pending outbox row.
    """
    from hermes_cli.kanban_db import _enqueue_receipt_outbox

    receipt_file = tmp_path / "c20-rtt.txt"
    receipt_file.write_text("rtt probe fixture\n")
    result_text = (
        "VERIFIED: kanban-rtt-probe round-trip against "
        f"{receipt_file}"
    )
    conn = _seed_db(
        tmp_path,
        task_id="t_c20rtt01",
        result_text=result_text,
        assignee=None,
        workspace_path=str(tmp_path),
    )

    digest = _enqueue_receipt_outbox(
        conn,
        task_id="t_c20rtt01",
        run_id=3,
        summary=result_text,
        result=result_text,
        metadata={},
        completed_at=int(time.time()),
    )

    rejects = conn.execute(
        "SELECT COUNT(*) c FROM task_events WHERE kind='receipt_outbox_rejected'"
    ).fetchone()["c"]
    row = conn.execute(
        "SELECT payload_json FROM kanban_receipt_outbox WHERE task_id=?",
        ("t_c20rtt01",),
    ).fetchone()

    assert digest is not None
    assert rejects == 0
    assert row is not None
    payload = json.loads(row["payload_json"])
    assert payload["assignee"] == "system"
    conn.close()
