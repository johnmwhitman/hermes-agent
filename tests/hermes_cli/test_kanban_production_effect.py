"""Tests for the installed production_effect module (hermes_cli.kanban_production_effect).

Applied by platformops (t_066aaaae) on 2026-09-14 from the design at
strategy/fleet-reorientation-20260913/PRODUCTION-EFFECT-FIELD.md.

Does not open the live kanban.db — all DB tests use in-memory sqlite.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# CLI / dispatcher path gate (t_3b87204e)
# ---------------------------------------------------------------------------


@pytest.fixture
def _prod_effect_db(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a kanban DB that has the production_effect column."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb.init_db()
    with kbc.connect_closing() as conn:
        from hermes_cli.kanban_production_effect import migrate_add_column
        migrate_add_column(conn)
    return home


def test_cli_complete_production_card_refused_without_deploy_evidence(_prod_effect_db):
    """``hermes kanban complete <production card>`` must refuse when no
    deploy evidence is provided — the gate now lives in ``complete_task``
    itself, not only in the agent tool path.

    This is the regression test for t_3b87204e: before the fix, the CLI
    path (``hermes_cli/kanban.py:_cmd_complete``) called
    ``kb.complete_task`` directly with no production_effect check, so a
    production card could be closed from the CLI without any deploy
    artifact.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        # Simulate a running task so complete_task accepts the transition.
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        ok = kb.complete_task(
            conn, tid,
            result="shipped to production",
            summary="shipped to production",
        )
    assert not ok, (
        "complete_task must refuse a production card with no deploy evidence; "
        "this is the Q25 gate bypass (t_3b87204e)."
    )


def test_cli_complete_production_card_accepted_with_deploy_evidence(_prod_effect_db):
    """When deploy evidence IS provided, the CLI path must succeed."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        ok = kb.complete_task(
            conn, tid,
            result="VERIFIED\ndeploy_id: dpl_abc123\ngit log --oneline -1",
            summary="deploy_id: dpl_abc123",
            metadata={"production_artifact": {"kind": "deploy_id", "value": "dpl_abc123"}},
        )
    assert ok, "complete_task must accept a production card with valid deploy evidence."


def test_cli_complete_internal_card_unaffected(_prod_effect_db):
    """Internal cards (NULL/internal production_effect) must pass the gate unchanged."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="internal task", assignee="dev")
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        ok = kb.complete_task(
            conn, tid,
            result="VERIFIED\ndone with internal work\ngit log --oneline -1",
            summary="internal task completed",
        )
    assert ok, "complete_task must accept internal cards without deploy evidence."


def test_cli_complete_enabling_card_refused_without_unblocks(_prod_effect_db):
    """An enabling card must name the production card it unblocks."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="enabling task", assignee="dev", production_effect="enabling")
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        ok = kb.complete_task(
            conn, tid,
            result="VERIFIED\nwired the token\ngit log --oneline -1",
            summary="enabling task done",
        )
    assert not ok, "complete_task must refuse an enabling card with no unblocks reference."


def test_complete_task_emits_blocked_audit_event(_prod_effect_db):
    """The gate must leave an auditable ``completion_blocked_production_effect``
    event so operators can trace refusals even when the caller swallows the
    receipt (dispatcher salvage path, scripts, batch completers). Without
    this event the only evidence of a refusal would be a log line.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        kb.complete_task(
            conn, tid,
            result="shipped without evidence",
            summary="shipped without evidence",
        )

    with kbc.connect_closing() as conn:
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "ORDER BY id",
            (tid,),
        ).fetchall()

    blocked = [e for e in events if e["kind"] == "completion_blocked_production_effect"]
    assert len(blocked) == 1, f"expected exactly one blocked event, got {events!r}"
    payload = json.loads(blocked[0]["payload"])
    assert payload["effect"] == "production"
    assert payload["classification"] == "missing"
    assert "production artifact" in payload["detail"]


def test_complete_task_raises_when_opted_in(_prod_effect_db):
    """Callers that want the structured receipt (CLI, tool) pass
    ``raise_on_production_effect=True`` and get a
    :class:`kb.ProductionEffectError` carrying the receipt dict.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        with pytest.raises(kb.ProductionEffectError) as excinfo:
            kb.complete_task(
                conn, tid,
                result="shipped",
                summary="shipped",
                raise_on_production_effect=True,
            )
    receipt = excinfo.value.receipt
    assert receipt["ok"] is False
    assert receipt["effect"] == "production"
    assert receipt["classification"] == "missing"


def test_cli_complete_surfaces_production_receipt(_prod_effect_db, capsys):
    """``hermes kanban complete <production card>`` must print the structured
    production_effect refusal so the operator can fix the receipt. Without
    the integration test the CLI path could regress to a generic
    'cannot complete (unknown id or terminal state)' even when
    ``complete_task`` gates correctly.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        kb.claim_task(conn, tid)

    from hermes_cli import kanban as kanban_cli
    args = argparse.Namespace(
        task_ids=[tid], summary=None, metadata=None, result="shipped",
    )
    rc = kanban_cli._cmd_complete(args)
    assert rc != 0, f"CLI must exit non-zero on refusal, got rc={rc}"
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "production_effect" in combined, (
        f"CLI must surface the gate name in its error channel; got: {captured!r}"
    )
    assert "production artifact" in combined, (
        f"CLI must surface the gate's detail; got: {captured!r}"
    )


def test_tool_complete_surfaces_production_receipt(_prod_effect_db, monkeypatch, tmp_path):
    """The agent tool path (``tools.kanban_tools._handle_complete``) must
    keep its existing user-facing ``tool_error`` shape after the gate moved
    from the tool into ``complete_task``. The tool now opts into the
    exception via ``raise_on_production_effect=True``.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        kb.claim_task(conn, tid)

    # Worker-env stubs the tool requires (worker session + run id).
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "dev")

    # Use a receipt-shaped summary + a real artifact path so the upstream
    # ``_enforce_receipt_on_complete`` gate does not pre-empt our test of
    # the production_effect gate. The summary must NOT contain
    # ``deploy_id:``, ``release_tag:``, or ``live_url:`` tokens — those
    # satisfy the production_effect gate.
    artifact = tmp_path / "deploy.log"
    artifact.write_text("released\n")
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": f"VERIFIED done shipped to production\n$ cat {artifact}",
        "result": f"VERIFIED done shipped to production\n$ cat {artifact}",
        "metadata": {},
    })
    parsed = json.loads(out)
    assert "error" in parsed, f"expected refusal, got: {parsed}"
    err = parsed["error"]
    assert "production_effect" in err, f"expected production_effect in error, got: {err!r}"
    assert "production" in err
    assert "missing" in err  # classification

    # Confirm the card was NOT moved to done (gate ran, did not silently pass).
    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] != "done", f"card must remain non-done after refusal; status={row['status']!r}"


def test_dispatcher_salvage_path_gated(_prod_effect_db):
    """The dispatcher salvage path in ``kanban_db_dispatch`` calls
    ``_kb.complete_task`` directly; with the gate in ``complete_task`` it
    is now gated too. A production card with no deploy evidence must NOT
    be salvage-completable.

    This is the third caller — without the gate in ``complete_task``
    itself, a dispatcher salvage of a production card could bypass
    the deploy-evidence requirement.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="prod deploy", assignee="dev", production_effect="production")
        kb.claim_task(conn, tid)

    with kbc.connect_closing() as conn:
        # complete_task default (raise_on_production_effect=False) returns False.
        ok = kb.complete_task(
            conn, tid,
            result="salvaged_clean_exit run_id=1 source=salvage",
            summary="salvaged_clean_exit run_id=1 source=salvage",
            metadata={"source": "salvage", "salvaged_clean_exit": True, "salvaged_run_id": 1},
        )
    assert not ok, "salvage path must NOT close a production card with no deploy evidence."

    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] != "done", "card must remain non-done after salvage-time gate refusal."


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
