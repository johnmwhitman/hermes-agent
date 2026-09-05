"""Tests for the body-receipt-clause gate extension on tools/kanban_tools.py.

Card t_2db4f45a: harden kanban_complete so a card whose body carries a
``receipt:`` block (the seat's authoritive "this is what success looks
like" clause) cannot land ``done`` with a count that does not match.

Sample body shape (from the 2026-09-05 seat cards):

    receipt:
      check: sqlite3 -readonly <db> < <sql>
      expected: 0 — 24 now (the largest of the five unchanged counts...)

The kernel parses the clause, re-runs the check on the done transition,
parses the FIRST integer in stdout (or ``rc N`` in the expected line),
and refuses the close when the observed integer is not equal. Empty
``result`` text on a receipted card is refused outright. Cards without
the clause keep the old prose / path / SHA / attachment evidence path
unchanged.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional, Tuple

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
    """Stand-in for a sqlite3 connection. Default has no attachments."""

    def __init__(self, rows=None):
        self._rows = rows or []

    def execute(self, *a, **kw):
        return _FakeCursor(self._rows)


def _body_with_receipt(check: str, expected: str) -> str:
    """Build a minimal card body that contains a receipt: clause."""
    return textwrap.dedent(
        f"""
        ## ASK
        Do the thing.

        receipt:
          check: {check}
          expected: {expected}
        """
    ).lstrip("\n")


# ---------------------------------------------------------------------------
# Clause parsing — direct unit tests on the helper
# ---------------------------------------------------------------------------


def test_parse_receipt_clause_basic():
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "echo 0",
        "0 — 24 now (the largest of the five unchanged counts)",
    )
    parsed = kt._parse_body_receipt_clause(body)
    assert parsed is not None
    check, expected, is_rc = parsed
    assert check == "echo 0"
    assert expected == 0
    assert is_rc is False


def test_parse_receipt_clause_rc_style_expected():
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "sh /Users/johnwhitman/AI/agents/runbooks/checks/kanban-offsite.sh",
        "rc 0 (newest COMPLETE set has ...) — RED now",
    )
    parsed = kt._parse_body_receipt_clause(body)
    assert parsed is not None
    check, expected, is_rc = parsed
    assert check == "sh /Users/johnwhitman/AI/agents/runbooks/checks/kanban-offsite.sh"
    assert expected == 0
    assert is_rc is True


def test_parse_receipt_clause_absent_returns_none():
    from tools import kanban_tools as kt

    body = "Just a regular body, no receipt: block at all."
    assert kt._parse_body_receipt_clause(body) is None


def test_parse_receipt_clause_ignores_non_receipt_markdown_headers():
    """A line that says ``receipt:`` inside a code fence must NOT be parsed
    as the clause — it's documentation. Cheap precaution."""
    from tools import kanban_tools as kt

    body = (
        "# How cards work\n\n"
        "Cards carry a YAML-ish `receipt:` block.\n\n"
        "```\n"
        "receipt:\n"
        "  check: echo secret\n"
        "  expected: 0\n"
        "```\n"
    )
    # The keyword is in a fenced code block — refuse to parse.
    assert kt._parse_body_receipt_clause(body) is None


# ---------------------------------------------------------------------------
# Gate behaviour — passing the clause
# ---------------------------------------------------------------------------


def test_gate_accepts_body_clause_when_check_matches(tmp_path):
    """Body has receipt: clause, check runs and prints 0 (== expected) → permit."""
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "printf '%s\\n' 0",
        "0 — RED now",
    )
    err = kt._enforce_receipt_on_complete(
        {
            "summary": "VERIFIED on body-clause gate",
            "result": "printf '%s\\n' 0",
        },
        _FakeConn(),
        "t_clause_ok",
        body=body,
    )
    assert err is None, f"expected None, got {err!r}"


def test_gate_accepts_body_clause_rc_style(tmp_path):
    """``expected: rc 0`` shape — check exit status is 0, stdout prints 0
    anyway — still passes."""
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "true",
        "rc 0 (something) — RED now",
    )
    err = kt._enforce_receipt_on_complete(
        {"summary": "rc-true", "result": "true"},
        _FakeConn(),
        "t_clause_rc_ok",
        body=body,
    )
    assert err is None, f"expected None, got {err!r}"


# ---------------------------------------------------------------------------
# Gate behaviour — refusing the closure
# ---------------------------------------------------------------------------


def test_gate_refuses_body_clause_when_check_count_mismatch():
    """Body has receipt: clause, check prints 24 (expected 0) → REFUSE."""
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "printf '%s\\n' 24",
        "0 — 24 now",
    )
    err = kt._enforce_receipt_on_complete(
        {
            "summary": "DID: ran sql, count dropped from 24 to 0",
            "result": "VERIFIED receipt",
        },
        _FakeConn(),
        "t_clause_fail",
        body=body,
    )
    assert err is not None, "expected refusal, got None"
    parsed = json.loads(err)
    assert "refused" in parsed["error"]
    # The actual observed count is in the message so the worker can act.
    assert "24" in parsed["error"] or "0" in parsed["error"]


def test_gate_refuses_body_clause_when_check_exits_nonzero():
    """check exits with rc=2 (the audit's literal failure mode) → REFUSE."""
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "sh -c 'echo broken; exit 2'",
        "0",
    )
    err = kt._enforce_receipt_on_complete(
        {"summary": "claim success", "result": "rc=0"},
        _FakeConn(),
        "t_clause_nonzero",
        body=body,
    )
    assert err is not None, "expected refusal, got None"
    parsed = json.loads(err)
    assert "refused" in parsed["error"]


def test_gate_refuses_empty_result_on_receipted_card():
    """Empty result on a receipted card → REFUSE outright (the second
    half of the seat's ask)."""
    from tools import kanban_tools as kt

    body = _body_with_receipt("printf '0\\n'", "0")
    err = kt._enforce_receipt_on_complete(
        {
            # Existing path-style evidence would normally pass the gate,
            # but on a receipted card the empty-result clause takes
            # precedence — refuse.
            "summary": "wrote /tmp/anything",
            "result": "",
            "artifacts": [str(Path("/tmp/anything"))],
        },
        _FakeConn(),
        "t_clause_empty_result",
        body=body,
    )
    assert err is not None, "expected refusal, got None"
    parsed = json.loads(err)
    assert "refused" in parsed["error"]
    assert "empty" in parsed["error"].lower() or "result" in parsed["error"].lower()


def test_gate_refuses_when_only_summary_no_result():
    """A receipted card where the worker passed summary but no result
    prose at all (and the body's check would have matched) is still
    refused — the empty-result clause runs first."""
    from tools import kanban_tools as kt

    body = _body_with_receipt("printf '0\\n'", "0")
    err = kt._enforce_receipt_on_complete(
        {
            "summary": "",
            "result": "",
        },
        _FakeConn(),
        "t_clause_no_result",
        body=body,
    )
    assert err is not None, "expected refusal, got None"
    parsed = json.loads(err)
    assert "refused" in parsed["error"]


def test_gate_accepts_rc_form_when_stdout_has_unrelated_int(tmp_path):
    """The kernel bug fixed by t_05647e4c:

    Body: ``expected: rc 0 (something — ...check passed)``
    Check: a script whose stdout carries a non-zero count **and** exits
    rc 0 (every seat shell that prints "N items, rc=0" looks like this).

    Pre-fix runner: parses ``1`` from stdout → mismatch → REFUSE.
    Post-fix runner: parses ``expected_is_rc=True`` from the body,
    compares against the process return code → ``0 == 0`` → permit.

    This is the shape the seat uses today:
      receipt:
        check: sh .../governor-sees-private-tmp.sh
        expected: rc 0 (governor items under /private/tmp: 1)
    """
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "printf 'governor items under /private/tmp: 1\\n'",
        "rc 0 (governor items under /private/tmp: 1)",
    )
    err = kt._enforce_receipt_on_complete(
        {
            "summary": "rc-form receipt carries rc=0",
            "result": (
                "Body check printed stdout `governor items under "
                "/private/tmp: 1` AND exited rc=0; the kernel honored "
                "the rc-form (expected_is_rc=True) and compared against "
                "the return code, not the unrelated stdout count."
            ),
        },
        _FakeConn(),
        "t_clause_rc_with_stdout_count",
        body=body,
    )
    assert err is None, f"expected None, got {err!r}"


def test_gate_refuses_rc_form_when_actual_rc_is_nonzero(tmp_path):
    """Mirror of the previous test: same body shape, but the check
    exits rc=2. The runner must observe rc=2 and refuse — never
    silently accept the rc=0 fallback because stdout had no integer."""
    from tools import kanban_tools as kt

    body = _body_with_receipt(
        "sh -c 'echo 7 items; exit 2'",
        "rc 0",
    )
    err = kt._enforce_receipt_on_complete(
        {
            "summary": "claim rc=0",
            "result": "I claim success but the check actually exited 2",
        },
        _FakeConn(),
        "t_clause_rc_nonzero_with_stdout_count",
        body=body,
    )
    assert err is not None, "expected refusal, got None"
    parsed = json.loads(err)
    assert "refused" in parsed["error"]


def test_gate_accepts_nonempty_result_on_receipted_card_matching_clause():
    """Body has receipt + body check matches + worker provides non-empty
    result prose describing what they observed → permit (existing
    evidence rules don't apply; the body's check is the verdict)."""
    from tools import kanban_tools as kt

    body = _body_with_receipt("printf '0\\n'", "0")
    err = kt._enforce_receipt_on_complete(
        {
            "summary": (
                "DID: cleared all 24 blocked cards in overwatch lane. "
                "VERIFIED receipt on body's clause"
            ),
            "result": (
                "Body check `printf 0` returned 0 (= expected); the "
                "previously blocking cards are now done or retyped."
            ),
        },
        _FakeConn(),
        "t_clause_ok_with_prose",
        body=body,
    )
    assert err is None, f"expected None, got {err!r}"


def test_gate_refuses_hollow_prose_when_receipted_card_check_fails():
    """A receipted card whose body's check FAILS → REFUSE, regardless of
    whether the result prose had a VERIFIED+cmd fragment (the body
    clause verdict, not the prose verdict, is the gate's first
    question)."""
    from tools import kanban_tools as kt

    body = _body_with_receipt("printf '7\\n'", "0")
    err = kt._enforce_receipt_on_complete(
        {
            # Even with a strong-looking result, the body check fails.
            "summary": "",
            "result": "VERIFIED\n$ echo done",
        },
        _FakeConn(),
        "t_clause_hollow_check_fails",
        body=body,
    )
    assert err is not None, "expected refusal, got None"
    parsed = json.loads(err)
    assert "refused" in parsed["error"]
    assert "7" in parsed["error"] or "observed" in parsed["error"]


# ---------------------------------------------------------------------------
# Gate behaviour — backward compatibility for cards WITHOUT the clause
# ---------------------------------------------------------------------------


def test_gate_unchanged_for_cards_without_receipt_clause(tmp_path):
    """Cards without a ``receipt:`` block keep the existing
    prose / path / SHA / attachment evidence path unchanged."""
    from tools import kanban_tools as kt

    artifact = tmp_path / "real.txt"
    artifact.write_text("x")
    err = kt._enforce_receipt_on_complete(
        {"summary": f"wrote {artifact}", "result": ""},
        _FakeConn(),
        "t_no_clause",
        body="this body has no clause at all, just prose.",
    )
    assert err is None


def test_gate_existing_behavior_preserved_for_empty_body_long_prose():
    """Cards without a body clause + long prose currently passes when
    the prose carries a VERIFIED + cmd fragment. Verify nothing
    regressed here."""
    from tools import kanban_tools as kt

    err = kt._enforce_receipt_on_complete(
        {
            "summary": "",
            "result": (
                "VERIFIED\n$ python3 -m pytest tests/ -q\n"
                "Tests passed cleanly; no flakes."
            ),
        },
        _FakeConn(),
        "t_no_clause_prose_ok",
        body="",
    )
    assert err is None
