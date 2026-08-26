"""Tests for the kanban_complete receipt gate (tools/kanban_tools.py).

Card t_948219e1: hard-gate kanban_complete so a card cannot move to
done unless an observable receipt is present. Sibling of
profiles/conductor/scripts/kanban_completion_verifier.py — same rules,
different layer. These tests exercise the gate function directly
without spinning up a real worker fixture; the e2e `_handle_complete`
rejections live next to the existing handler tests in
test_kanban_tools.py.
"""
from __future__ import annotations

import json
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, p):
        self._p = p

    def __getitem__(self, i):
        return self._p


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Stand-in for a sqlite3 connection that returns the given rows for
    the attachments query. Default is no attachments."""

    def __init__(self, rows=None):
        self._rows = rows or []

    def execute(self, *a, **kw):
        return _FakeCursor(self._rows)


# ---------------------------------------------------------------------------
# Gating unit tests — refuse
# ---------------------------------------------------------------------------


def test_gate_rejects_empty_summary():
    """result/summary that is empty prose → refuse the close."""
    from tools import kanban_tools as kt

    err = kt._enforce_receipt_on_complete(
        {"summary": "", "result": ""}, _FakeConn(), "t_doesnotexist",
    )
    assert err is not None
    parsed = json.loads(err)
    assert "refused" in parsed["error"]
    assert "Card not moved to done" in parsed["error"]


def test_gate_rejects_hollow_short_prose():
    """A short, plain-English summary with no path/SHA/VERIFIED → refuse."""
    from tools import kanban_tools as kt

    err = kt._enforce_receipt_on_complete(
        {"summary": "done", "result": ""}, _FakeConn(), "t_doesnotexist",
    )
    assert err is not None
    parsed = json.loads(err)
    assert "refused" in parsed["error"]


def test_gate_rejects_long_prose_without_receipt():
    """A long result with no path/SHA/VERIFIED → still refuse (matches the
    after-the-fact verifier's verdict_for() rule)."""
    from tools import kanban_tools as kt

    result = (
        "I did the work and updated the file and the config and "
        "the docs and the tests, all in a single clean pass."
    )  # > 80 chars
    err = kt._enforce_receipt_on_complete(
        {"summary": "", "result": result}, _FakeConn(), "t_doesnotexist",
    )
    assert err is not None
    parsed = json.loads(err)
    assert "no observable receipt" in parsed["error"]


def test_gate_rejects_unresolvable_path():
    """A path that doesn't exist on disk → refuse (unobservable)."""
    from tools import kanban_tools as kt

    err = kt._enforce_receipt_on_complete(
        {
            "summary": "see the file",
            "result": "see /tmp/kanban-receipt-gate-does-not-exist-xyzzy.txt",
        },
        _FakeConn(),
        "t_doesnotexist",
    )
    assert err is not None
    parsed = json.loads(err)
    assert "do not exist on disk" in parsed["error"]
    assert (
        "/tmp/kanban-receipt-gate-does-not-exist-xyzzy.txt" in parsed["error"]
    )


# ---------------------------------------------------------------------------
# Gating unit tests — permit
# ---------------------------------------------------------------------------


def test_gate_accepts_existing_path(tmp_path):
    """A receipt text naming an existing absolute path → permit."""
    from tools import kanban_tools as kt

    artifact = tmp_path / "real.txt"
    artifact.write_text("x")
    err = kt._enforce_receipt_on_complete(
        {"summary": f"wrote {artifact}", "result": ""},
        _FakeConn(),
        "t_doesnotexist",
    )
    assert err is None


def test_gate_accepts_verified_plus_command():
    """VERIFIED token + a command-shaped fragment → permit (no real run).
    The verifier's CMD_RE only matches at the start of a line, so the
    conventional shape is VERIFIED on one line, $ cmd on the next."""
    from tools import kanban_tools as kt

    err = kt._enforce_receipt_on_complete(
        {
            "summary": "",
            "result": "VERIFIED\n$ python3 -m pytest tests/ -q",
        },
        _FakeConn(),
        "t_doesnotexist",
    )
    assert err is None


def test_gate_accepts_resolved_git_sha():
    """A real git SHA reachable in ~/AI → permit."""
    from pathlib import Path

    from tools import kanban_tools as kt

    ai = Path("/Users/johnwhitman/AI")
    try:
        sha = subprocess.run(
            ["git", "-C", str(ai), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception:
        pytest.skip("no git in ~/AI")
    if not sha or len(sha) < 10:
        pytest.skip("no resolvable HEAD")
    err = kt._enforce_receipt_on_complete(
        {"summary": f"shipped {sha[:12]}", "result": ""},
        _FakeConn(),
        "t_doesnotexist",
    )
    assert err is None


def test_gate_accepts_existing_attachment(tmp_path):
    """An attachment whose stored_path still exists on disk → permit."""
    from tools import kanban_tools as kt

    attach_path = tmp_path / "attach.bin"
    attach_path.write_text("data")
    conn = _FakeConn(rows=[_FakeRow(str(attach_path))])
    err = kt._enforce_receipt_on_complete(
        {"summary": "done", "result": ""},
        conn,
        "t_doesnotexist",
    )
    assert err is None


def test_gate_rejects_attachment_that_disappeared(tmp_path):
    """An attachment row whose file no longer exists must not count."""
    from tools import kanban_tools as kt

    dead = tmp_path / "missing.bin"  # never created
    conn = _FakeConn(rows=[_FakeRow(str(dead))])
    err = kt._enforce_receipt_on_complete(
        {
            "summary": "done",
            "result": "no other receipt; relying on attachment",
        },
        conn,
        "t_doesnotexist",
    )
    assert err is not None
    parsed = json.loads(err)
    assert "refused" in parsed["error"]
    # The exact message depends on the prose length: short "result" hits
    # the "too short" branch; either is a refusal. Make sure we did NOT
    # accept the (missing) attachment as a receipt.
    assert "Card not moved to done" in parsed["error"]
