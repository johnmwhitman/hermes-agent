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


# ---------------------------------------------------------------------------
# Multi-repo SHA resolver (t_cca67626)
# ---------------------------------------------------------------------------
#
# The card body's SHA resolver used to be hard-coded to /Users/johnwhitman/AI.
# Any receipt SHA living in another repo (hermes-agent, a lane product repo,
# a scratch worktree) was silently dropped — exactly the bug that nuked
# 106 completions' outbox rows. These tests pin the new contract: the
# resolver consults the supplied candidate repos in order and treats the
# SHA as resolvable iff any one of them claims it.
# --------------------------------------------------------------------------


def test_receipt_classify_multi_repo_accepts_external_sha(tmp_path):
    """A SHA living in a non-~/AI repo resolves when the candidate list
    includes that repo. Mirrors the t_ae9975c9 production failure."""
    import subprocess
    from pathlib import Path

    from tools import kanban_tools as kt

    repo = tmp_path / "ext-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "--initial-branch", "main"], cwd=str(repo), check=True)
    # Configure identity so the commit does not fail on CI hosts.
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"], check=True,
    )
    (repo / "f.txt").write_text("hi")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    assert len(sha) >= 10
    # Verify the SHA does NOT resolve in the production default — that's
    # the literal bug surface.
    assert not kt._receipt_sha_resolves(sha, Path("/Users/johnwhitman/AI"))
    # But the candidate list accepts it.
    assert kt._receipt_sha_resolves_in(sha, [repo, Path("/Users/johnwhitman/AI")])
    # And the classifier carries it through to ``existing_shas``.
    ev = kt._receipt_classify(
        f"VERIFIED commit {sha}", [], repos=[repo, Path("/Users/johnwhitman/AI")],
    )
    assert sha in ev["existing_shas"], ev
    assert ev["has_verified"] is True


def test_receipt_classify_default_repo_unchanged_for_backwards_compat():
    """When ``repos`` is omitted, the classifier still falls back to
    /Users/johnwhitman/AI (the historical default) so legacy callers
    behave exactly as before."""
    from pathlib import Path

    from tools import kanban_tools as kt

    # No repos arg → default ~/AI single-root resolution (existing test
    # `test_gate_accepts_resolved_git_sha` exercises the green path; this
    # test pins the resolver contract from the classifier side).
    ev = kt._receipt_classify("VERIFIED nothing resolves here", [])
    assert ev["existing_shas"] == []
    assert ev["has_verified"] is True
    assert ev["has_cmd"] is False


def test_receipt_repo_candidates_orders_workspace_then_fallback(tmp_path):
    """``_receipt_repo_candidates`` returns the workspace repo first and
    the historical ~/AI fallback last, with duplicates removed."""
    from pathlib import Path

    from tools import kanban_tools as kt

    ws = tmp_path / "ws"
    ws.mkdir()
    out = kt._receipt_repo_candidates(str(ws), project_repo=None)
    # The workspace path is not inside a git repo, so the helper falls
    # through to the fallback. The candidate list still has the literal
    # workspace path AND the fallback, in that order.
    assert out[0] == ws
    assert Path("/Users/johnwhitman/AI") in out
    # Fallback is always last.
    assert out[-1] == Path("/Users/johnwhitman/AI")


def test_receipt_repo_candidates_dedups_overlap(tmp_path):
    """When workspace and fallback resolve to the same path, the candidate
    list contains it exactly once."""
    from pathlib import Path

    from tools import kanban_tools as kt

    # Pass /Users/johnwhitman/AI as the workspace AND as the fallback.
    # The helper should de-dupe and return a single-element list.
    out = kt._receipt_repo_candidates(
        "/Users/johnwhitman/AI", project_repo=None,
        fallback=Path("/Users/johnwhitman/AI"),
    )
    assert out == [Path("/Users/johnwhitman/AI")]
    assert len(out) == len(set(str(p) for p in out))


def test_receipt_repo_candidates_resolves_git_top_level(tmp_path):
    """When ``workspace_path`` is a subdir of a git repo (worktree
    layout), the candidate list contains the repo's top level so the
    SHA resolver sees the right git-dir."""
    import subprocess
    from pathlib import Path

    from tools import kanban_tools as kt

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "--initial-branch", "main"], cwd=str(repo), check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "x@x"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "x"], check=True)
    (repo / "f").write_text("y")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "i"], check=True)
    sub = repo / "sub" / "deeper"
    sub.mkdir(parents=True)
    out = kt._receipt_repo_candidates(str(sub), project_repo=None)
    # Top-level comes back, not the subdir.
    assert Path(repo).resolve() == out[0].resolve()
    assert out[0] != sub
