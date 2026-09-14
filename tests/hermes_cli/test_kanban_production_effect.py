"""Tests for the installed production_effect module (hermes_cli.kanban_production_effect).

Applied by platformops (t_066aaaae) on 2026-09-14 from the design at
strategy/fleet-reorientation-20260913/PRODUCTION-EFFECT-FIELD.md.

Does not open the live kanban.db — all DB tests use in-memory sqlite.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from hermes_cli.kanban_production_effect import (  # noqa: E402
    classify_effect,
    column_present,
    count_effects,
    enforce_on_complete,
    format_cycle_summary_line,
    migrate_add_column,
    parse_production_artifact,
    parse_unblocks,
    phase05_published,
    validate_production_effect,
)


# ---------------------------------------------------------------------------
# validate / classify
# ---------------------------------------------------------------------------


def test_validate_none_and_blank_are_null():
    assert validate_production_effect(None) is None
    assert validate_production_effect("") is None
    assert validate_production_effect("  ") is None


def test_validate_accepts_three_literals():
    assert validate_production_effect("production") == "production"
    assert validate_production_effect("Enabling") == "enabling"
    assert validate_production_effect("INTERNAL") == "internal"


def test_validate_rejects_unknown():
    try:
        validate_production_effect("user-facing")
    except ValueError as exc:
        assert "production|enabling|internal" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_classify_null_and_garbage_are_internal():
    assert classify_effect(None) == "internal"
    assert classify_effect("") == "internal"
    assert classify_effect("user-facing") == "internal"


# ---------------------------------------------------------------------------
# kill switch — old path
# ---------------------------------------------------------------------------


def test_kill_switch_off_skips_even_production():
    receipt = enforce_on_complete(
        "production",
        summary="shipped nothing",
        environ={"HERMES_KANBAN_PRODUCTION_EFFECT": "off"},
    )
    assert receipt["ok"] is True
    assert receipt["classification"] == "disabled"
    assert receipt["effect"] == "production"


def test_internal_is_always_old_path():
    receipt = enforce_on_complete("internal", summary="done")
    assert receipt["ok"] is True
    assert receipt["classification"] == "skipped"
    receipt_null = enforce_on_complete(None, summary="done")
    assert receipt_null["ok"] is True
    assert receipt_null["effect"] == "internal"


# ---------------------------------------------------------------------------
# production gate
# ---------------------------------------------------------------------------


def test_production_refuses_without_artifact():
    receipt = enforce_on_complete(
        "production",
        summary="VERIFIED\n$ pytest -q",
        result="/tmp/exists.txt",
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "missing"
    assert "production artifact" in receipt["detail"]


def test_production_accepts_deploy_id_in_metadata():
    receipt = enforce_on_complete(
        "production",
        summary="shipped",
        metadata={"production_artifact": {"kind": "deploy_id", "value": "dpl_abc123"}},
    )
    assert receipt["ok"] is True
    assert receipt["classification"] == "deploy_id"
    assert receipt["artifact"]["value"] == "dpl_abc123"


def test_production_accepts_release_tag_in_prose():
    receipt = enforce_on_complete(
        "production",
        summary="release_tag: v1.2.3 landed on origin/main",
    )
    assert receipt["ok"] is True
    assert receipt["classification"] == "release_tag"
    assert receipt["artifact"]["value"] == "v1.2.3"


def test_production_live_url_requires_http_status():
    missing = enforce_on_complete(
        "production",
        summary="live_url: https://example.com/app",
    )
    assert missing["ok"] is False
    assert missing["classification"] == "invalid"
    ok = enforce_on_complete(
        "production",
        summary="live_url: https://example.com/app HTTP 200",
    )
    assert ok["ok"] is True
    assert ok["classification"] == "live_url"
    assert ok["artifact"]["observed_status"] == 200


def test_production_phase05_receipt_must_be_published():
    envelope = {
        "schema": "phase05-receipt/v1",
        "cycle_id": "t_abcdef01",
        "cycle_summary": "shipped the landing page copy",
        "issued_at": "2026-09-13T23:00:00Z",
        "issued_by": "yourbrief",
        "files_touched": [],
        "git": {
            "branch": "main",
            "tip_sha": "a" * 40,
            "raid_tip_before": "b" * 40,
            "raid_tip_after": "a" * 40,
            "ff_safe": True,
        },
        "gates": {"tsc": {"rc": 0, "stderr_sha256": "c" * 64, "exit_message": "EMPTY"},
                  "lint": {"rc": 0, "message_sha256": "d" * 64},
                  "test": {"rc": 0, "stdout_sha256": "e" * 64}},
        "key_sweep": {"hits": 0, "regex": "AKIA"},
        "craft": {
            "user_visible_change": "landing page now shows the new price",
            "smallest_measurable_improvement": "one live URL returns HTTP 200",
            "falsifier": "curl -sI https://example.com returns 5xx",
        },
        "co_signer": {
            "required": False, "peer": None, "attestation": None,
            "signed_at": None, "state": "N/A",
        },
        "state": "LOCAL_ONLY",
    }
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        local = tmp_path / "local.json"
        local.write_text(json.dumps(envelope))
        refused = enforce_on_complete(
            "production",
            metadata={"production_artifact": {"kind": "phase05_receipt", "value": str(local)}},
        )
        assert refused["ok"] is False
        envelope["state"] = "PUBLISHED"
        published = tmp_path / "published.json"
        published.write_text(json.dumps(envelope))
        accepted = enforce_on_complete(
            "production",
            artifacts=[str(published)],
            summary="published the cycle",
        )
        assert accepted["ok"] is True
        assert accepted["classification"] == "phase05_receipt"


# ---------------------------------------------------------------------------
# enabling gate
# ---------------------------------------------------------------------------


def test_enabling_refuses_without_unblocks():
    receipt = enforce_on_complete("enabling", summary="unblocked the queue")
    assert receipt["ok"] is False
    assert "unblocks" in receipt["detail"]


def test_enabling_accepts_named_production_card():
    def lookup(tid):
        if tid == "t_deadbeef01":
            return {"id": tid, "production_effect": "production"}
        return None

    receipt = enforce_on_complete(
        "enabling",
        summary="wired the deploy token",
        metadata={"unblocks": "t_deadbeef01"},
        lookup_task=lookup,
    )
    assert receipt["ok"] is True
    assert receipt["unblocks"] == "t_deadbeef01"
    assert receipt["classification"] == "unblocks"


def test_enabling_parses_prose_unblocks():
    def lookup(tid):
        return {"id": tid, "production_effect": "production"}

    receipt = enforce_on_complete(
        "enabling",
        result="unblocks: t_aabbccdd01 after the secret landed",
        lookup_task=lookup,
    )
    assert receipt["ok"] is True
    assert receipt["unblocks"] == "t_aabbccdd01"


def test_enabling_refuses_internal_target():
    def lookup(tid):
        return {"id": tid, "production_effect": "internal"}

    receipt = enforce_on_complete(
        "enabling",
        metadata={"unblocks": "t_deadbeef01"},
        lookup_task=lookup,
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "invalid"
    assert "not production" in receipt["detail"]


def test_enabling_refuses_missing_target():
    receipt = enforce_on_complete(
        "enabling",
        metadata={"unblocks": "t_deadbeef01"},
        lookup_task=lambda _tid: None,
    )
    assert receipt["ok"] is False
    assert "does not exist" in receipt["detail"]


# ---------------------------------------------------------------------------
# migration + counts (in-memory sqlite; never the live board)
# ---------------------------------------------------------------------------


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT)")
    conn.executemany(
        "INSERT INTO tasks VALUES (?, ?)",
        [("t_a", "done"), ("t_b", "ready"), ("t_c", "done")],
    )
    return conn


def test_missing_column_counts_everything_internal():
    conn = _fresh_conn()
    assert column_present(conn) is False
    assert count_effects(conn) == {"production": 0, "enabling": 0, "internal": 3}
    assert count_effects(conn, done_only=True) == {
        "production": 0, "enabling": 0, "internal": 2,
    }


def test_migrate_add_column_is_idempotent_and_defaults_null():
    conn = _fresh_conn()
    assert migrate_add_column(conn) is True
    assert column_present(conn) is True
    assert migrate_add_column(conn) is False
    nulls = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE production_effect IS NULL"
    ).fetchone()[0]
    assert nulls == 3
    assert count_effects(conn)["internal"] == 3


def test_counts_split_after_explicit_set():
    conn = _fresh_conn()
    migrate_add_column(conn)
    conn.execute("UPDATE tasks SET production_effect='production' WHERE id='t_a'")
    conn.execute("UPDATE tasks SET production_effect='enabling' WHERE id='t_b'")
    # t_c stays NULL → internal
    assert count_effects(conn) == {"production": 1, "enabling": 1, "internal": 1}
    assert count_effects(conn, done_only=True) == {
        "production": 1, "enabling": 0, "internal": 1,
    }


def test_cycle_summary_line_shape():
    line = format_cycle_summary_line(
        {"production": 4, "enabling": 10, "internal": 6000},
        {"production": 1, "enabling": 3, "internal": 3400},
    )
    assert line.startswith("  kanban_effect: ")
    assert "production=4/1" in line
    assert "enabling=10/3" in line
    assert "internal=6000/3400" in line
    assert "(all/done)" in line


def test_dashboard_helper_does_not_break_when_column_absent():
    """inventory_truth.kanban_task_counts stays status/assignee-only.

    The additive helper lives next to it; missing column must not raise.
    """
    conn = _fresh_conn()
    counts = count_effects(conn)
    assert set(counts) == {"production", "enabling", "internal"}


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 — report then fail the suite
            failed.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"RESULT pass={len(tests) - len(failed)} fail={len(failed)}")
    raise SystemExit(1 if failed else 0)
