"""Tests for kanban goal-loop transport-failure handling.

The kanban goal loop (``hermes_cli.goals.run_kanban_goal_loop``) drives a
goal_mode worker turn-by-turn, calling ``judge_goal`` after each turn to
decide whether the worker's response satisfies the card. The auxiliary
``judge_goal`` returns ``(verdict, reason, parse_failed, wait_directive,
transport_failed)``. A ``transport_failed`` flag means the judge could
not reach the API at all (auth 401, DNS, connection error).

Persistent goals (``GoalManager.evaluate_after_turn``) already have a
bounded consecutive-transport-failure pause pattern: after
``DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES`` (=5) consecutive
``transport_failed=True`` replies, the loop auto-pauses so a permanently
broken judge cannot burn the entire turn budget.

This file pins the contract for the kanban goal loop: it must apply the
same bounded pattern, preserve actual worker progress, and never falsely
mark the task complete or manufacture a work failure. The judge transport
itself is mocked — no live provider call is ever made.

Issue: t_22414872 — Last Out W4 ``t_eb180fbf`` and Wickhand W1.1
``t_b0033ab3`` ran substantive work and ran out of goal-turn budget only
because the auxiliary judge could not reach the upstream API for eight
consecutive verdict calls. The kanban goal loop currently ignores
``_transport_failed`` and just keeps feeding continuation prompts until
the budget is exhausted, falsely labeling the cards as exhausted even
though the judge route — not the worker — failed.
"""

from __future__ import annotations

import pytest

from hermes_cli import goals


# ---------------------------------------------------------------------------
# Judge script helper
# ---------------------------------------------------------------------------


def _patch_judge_with_transport(monkeypatch, verdicts):
    """Make ``judge_goal`` return a scripted sequence with explicit transport flags.

    Each entry is a 5-tuple matching ``judge_goal``'s contract:
    ``(verdict, reason, parse_failed, wait_directive, transport_failed)``.
    """
    seq = list(verdicts)

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        if not seq:
            # After script exhaustion, fall back to a benign done so the
            # caller never wedges.
            return ("done", "script exhausted", False, None, False)
        v, reason, parse_failed, wait, transport = seq.pop(0)
        return (v, reason, parse_failed, wait, transport)

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)


# ---------------------------------------------------------------------------
# Pre-fix regression: 8 transport failures burned 8/8 turns on the buggy
# code, falsely labeling the worker as "budget exhausted". The fix replaces
# this with a transport-specific blocked_transport outcome that fires at the
# threshold, NOT at the budget boundary.
# ---------------------------------------------------------------------------


def test_pre_fix_8_transport_failures_consume_all_turns_until_budget(monkeypatch):
    """REGRESSION PIN — proves the bug from t_22414872.

    With ``max_turns=8`` and 8 scripted transport failures, the loop MUST
    NOT silently walk all 8 turns in a row. The fix short-circuits at the
    transport threshold (``DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES``)
    with outcome ``blocked_transport`` and a reason that names the
    transport problem.
    """
    threshold = goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    transport_only = ("continue", "xAI HTTP 502 via routeplane", False, None, True)
    # Script enough failures to also exhaust the budget if the threshold
    # is ignored (so a buggy loop would still hit ``blocked_budget`` at 8).
    _patch_judge_with_transport(monkeypatch, [transport_only] * 12)
    block_calls = []
    turns = []

    result = goals.run_kanban_goal_loop(
        task_id="t_judge_down",
        goal_text="finish scene check",
        run_turn=lambda p: turns.append(p) or "noop response",
        task_status_fn=lambda: "running",
        block_fn=block_calls.append,
        first_response="first turn reply",
        max_turns=12,
    )

    # The new contract: threshold paused the loop well before the budget.
    assert result["outcome"] == "blocked_transport", (
        f"expected blocked_transport (threshold trip), got {result!r}"
    )
    # First turn consumed one unit (turns_used=1 entering the loop), then
    # ``threshold - 1`` more continuations ran before the threshold tripped
    # on the next judge call.
    assert result["turns_used"] == threshold, (
        f"expected threshold={threshold} turns_used, got {result['turns_used']}"
    )
    assert len(turns) == threshold - 1
    assert len(block_calls) == 1
    reason = block_calls[0]
    # Block reason must explain transport, not budget exhaustion.
    assert "judge" in reason.lower(), reason
    assert "transport" in reason.lower(), reason
    assert "turn budget" not in reason.lower(), (
        f"transport pause must NOT be mislabeled as turn-budget exhaustion: {reason!r}"
    )


# ---------------------------------------------------------------------------
# Fixed behavior: successful judge + genuine continue paths still work
# ---------------------------------------------------------------------------


def test_worker_self_completion_short_circuits(monkeypatch):
    """When the worker calls kanban_complete on its first turn, no judge is consulted."""
    judge_calls = {"n": 0}

    def _fake_judge(*a, **kw):
        judge_calls["n"] += 1
        return ("done", "ignored", False, None, False)

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)

    result = goals.run_kanban_goal_loop(
        task_id="t_already_done",
        goal_text="g",
        run_turn=lambda p: pytest.fail("must not run another turn"),
        task_status_fn=lambda: "done",
        block_fn=lambda r: pytest.fail("must not block"),
        first_response="completed in first turn",
    )

    assert result["outcome"] == "completed_by_worker"
    assert result["turns_used"] == 1
    assert judge_calls["n"] == 0, (
        "judge must not be called when worker already terminated"
    )


def test_genuine_continue_consumes_one_turn_and_continues(monkeypatch):
    """A non-transport 'continue' verdict consumes one turn and prompts again."""
    _patch_judge_with_transport(
        monkeypatch,
        [
            ("continue", "still working on acceptance criteria", False, None, False),
            ("continue", "almost there", False, None, False),
            ("done", "ok", False, None, False),
            # After "done" the loop nudges finalize; the worker doesn't
            # finalize, so a 2nd "done" triggers the judged-done block.
            ("done", "ok", False, None, False),
        ],
    )
    turns = []
    block_calls = []

    result = goals.run_kanban_goal_loop(
        task_id="t_cont",
        goal_text="g",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "running",
        block_fn=block_calls.append,
        first_response="r1",
    )

    # First judge: continue → run_turn (turns=2). Second judge: continue
    # → run_turn (turns=3). Third judge: done → finalize → run_turn
    # (turns=4). Fourth judge: done, nudged → block. Three run_turns total.
    assert len(turns) == 3
    assert result["outcome"] == "blocked_budget"
    assert "finalize nudge" in block_calls[0].lower()


def test_budget_exhaustion_uses_budget_reason_when_no_transport(monkeypatch):
    """Pure non-transport continuations trip the budget for the right reason."""
    _patch_judge_with_transport(
        monkeypatch,
        [("continue", "real work still in progress", False, None, False)] * 10,
    )
    block_calls = []

    result = goals.run_kanban_goal_loop(
        task_id="t_exhaust",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=block_calls.append,
        first_response="r1",
        max_turns=5,
    )

    assert result["outcome"] == "blocked_budget"
    assert result["turns_used"] == 5
    assert len(block_calls) == 1
    reason = block_calls[0]
    assert "transport" not in reason.lower(), (
        f"genuine budget exhaustion must not be mislabeled as transport: {reason!r}"
    )
    assert "turn budget" in reason.lower(), reason


def test_transport_threshold_pauses_before_budget_exhaustion(monkeypatch):
    """N consecutive transport failures pause the loop BEFORE the budget
    exhausts, with a transport-specific reason.
    """
    threshold = goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    transport_only = ("continue", "xAI 502", False, None, True)
    _patch_judge_with_transport(monkeypatch, [transport_only] * threshold)
    block_calls = []

    result = goals.run_kanban_goal_loop(
        task_id="t_threshold",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=block_calls.append,
        first_response="r1",
        max_turns=20,
    )

    assert result["outcome"] == "blocked_transport"
    assert result["turns_used"] == threshold
    assert len(block_calls) == 1
    reason = block_calls[0]
    assert "judge" in reason.lower()
    assert "transport" in reason.lower()
    assert "turn budget" not in reason.lower()


def test_transport_recovers_after_initial_failures(monkeypatch):
    """A single successful judge verdict resets the consecutive counter."""
    _patch_judge_with_transport(
        monkeypatch,
        [
            ("continue", "502", False, None, True),
            ("continue", "502", False, None, True),
            ("continue", "real progress: scene passes", False, None, False),
            ("continue", "502", False, None, True),
            ("continue", "502", False, None, True),
            ("continue", "502", False, None, True),
            ("done", "ok", False, None, False),
        ],
    )
    block_calls = []

    result = goals.run_kanban_goal_loop(
        task_id="t_recover",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=block_calls.append,
        first_response="r1",
    )

    # After the "real progress" verdict the transport counter resets, so we
    # never reach the threshold. The final "done" triggers the
    # finalize → judged-done block.
    assert result["outcome"] != "blocked_transport", (
        f"transport recovery should reset the counter, got: {result}"
    )
    assert len(block_calls) == 1
    assert "finalize nudge" in block_calls[0].lower()


def test_parse_failures_do_not_extend_transport_counter(monkeypatch):
    """parse_failed and transport_failed must NOT stack.

    Five parse_failures + N transport failures == N transport failures
    (parse_failed does NOT count toward the transport threshold).
    """
    threshold = goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    # 2 parse failures + (threshold+2) transport failures. Without
    # isolation, the counter would trip after `threshold-2` transport
    # calls; with isolation, it must take exactly `threshold` transport
    # failures in a row to trip.
    verdicts = [
        ("continue", "judge returned prose", True, None, False),
        ("continue", "judge returned prose", True, None, False),
    ]
    verdicts += [("continue", "502", False, None, True)] * (threshold + 2)
    _patch_judge_with_transport(monkeypatch, verdicts)
    block_calls = []

    result = goals.run_kanban_goal_loop(
        task_id="t_mixed",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=block_calls.append,
        first_response="r1",
        max_turns=20,
    )

    assert result["outcome"] == "blocked_transport"
    assert len(block_calls) == 1
    assert "transport" in block_calls[0].lower()
    # parse_failures don't count toward transport; reason should NOT
    # conflate the two.
    assert "parse" not in block_calls[0].lower(), (
        f"transport reason should not mention parse failures: {block_calls[0]!r}"
    )


def test_lifecycle_state_records_judge_loop_outcome(monkeypatch):
    """The fix must surface a stable outcome string the kanban dispatcher
    can route on without parsing free-text reasons.

    All current outcomes: completed_by_worker, review_requested_by_worker,
    changes_requested_by_reviewer, blocked_by_worker, blocked_budget,
    stopped. After the fix: blocked_transport joins that set.
    """
    # Quick smoke: drive a transport trip and confirm the new outcome string.
    threshold = goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    transport_only = ("continue", "xAI 502", False, None, True)
    _patch_judge_with_transport(monkeypatch, [transport_only] * (threshold + 1))

    result = goals.run_kanban_goal_loop(
        task_id="t_outcome",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=lambda r: None,
        first_response="r1",
        max_turns=20,
    )

    assert "blocked_transport" == result["outcome"], result


def test_no_live_provider_call(monkeypatch):
    """Sanity: the loop must never invoke a real provider; the mocked
    judge is the only binding observed. Just exercises the happy path so
    a refactor that bypasses the script and calls auxiliary_client would
    surface as a collection / import failure (not a network call here —
    that would require live network).
    """
    seen_calls = {"n": 0}

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        seen_calls["n"] += 1
        return ("continue", "still working", False, None, False)

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)

    # Force the loop to stop quickly: status changes to "done" after a
    # couple of turns. Use a counter.
    state = {"calls": 0}

    def _status():
        state["calls"] += 1
        if state["calls"] >= 2:
            return "done"
        return "running"

    goals.run_kanban_goal_loop(
        task_id="t_sanity",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=_status,
        block_fn=lambda r: pytest.fail(f"should not block: {r}"),
        first_response="r1",
    )

    assert seen_calls["n"] >= 1, "judge was called at least once"
    assert state["calls"] >= 2, "status check ran at least twice"


# ---------------------------------------------------------------------------
# Successful judge `done` followed by worker finalization (happy finalize path).
# The pre-existing tests only covered: (a) worker self-completes before
# judging, (b) judged-done + worker never finalizes → blocked_budget. The
# remaining gap is the path where the judge says "done", the worker is
# nudged to finalize, and on the next turn it actually calls
# kanban_complete. That path MUST return ``completed_by_worker`` with
# ``turns_used=2`` — proving transport-fix doesn't break the normal happy
# finalize flow.
# ---------------------------------------------------------------------------


def test_judged_done_followed_by_worker_finalize_completes(monkeypatch):
    """A successful judge ``done`` verdict followed by the worker calling
    ``kanban_complete`` on the next turn completes the task cleanly with
    ``outcome == "completed_by_worker"``.

    Sequence: status=running → judge returns ``("done", ..., transport=False)``
    → prompt is the finalize template, ``nudged_to_finalize=True``, run_turn
    is called. On the next iteration ``task_status_fn()`` returns ``"done"``
    (worker called kanban_complete) → loop exits with the existing
    completion branch.
    """
    state = {"calls": 0}

    def _status():
        state["calls"] += 1
        # First status check: running. Second status check (after the
        # finalize nudge turn): done (worker called kanban_complete).
        if state["calls"] >= 2:
            return "done"
        return "running"

    _patch_judge_with_transport(
        monkeypatch,
        [
            ("done", "all acceptance criteria met", False, None, False),
        ],
    )
    finalize_prompts = []
    block_calls = []

    result = goals.run_kanban_goal_loop(
        task_id="t_done_then_finalize",
        goal_text="g",
        run_turn=lambda p: (
            finalize_prompts.append(p) or "I am calling kanban_complete now"
        ),
        task_status_fn=_status,
        block_fn=block_calls.append,
        first_response="r1",
    )

    assert result["outcome"] == "completed_by_worker", result
    assert result["turns_used"] == 2, result
    assert len(finalize_prompts) == 1, (
        f"expected exactly one finalize nudge turn, got {len(finalize_prompts)}"
    )
    assert len(block_calls) == 0, (
        f"happy finalize must NOT trigger block_fn, got: {block_calls!r}"
    )


# ---------------------------------------------------------------------------
# Real kanban lifecycle persistence: a transport trip writes ``block_kind =
# 'transient'`` into the tasks table (via kanban_db.block_task) and the
# returned outcome string is the typed "blocked_transport" the dispatcher
# can route on. This proves the typed-lifecycle fix beyond just the
# returned string — a return-string assertion alone (the prior
# test_lifecycle_state_records_judge_loop_outcome) cannot distinguish a
# truthful typed block from a string-only side effect.
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home_for_goal_loop(tmp_path, monkeypatch):
    """Per-test kanban home with an empty SQLite board."""
    from pathlib import Path as _Path

    from hermes_cli import kanban_db as _kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    _kb.init_db()
    return home


def test_blocked_transport_persists_transient_kind_in_db(
    monkeypatch, kanban_home_for_goal_loop
):
    """Drives a transport trip through the real ``kanban_db.block_task`` and
    asserts ``tasks.block_kind == "transient"`` plus a ``blocked`` status
    in the on-disk board — the typed transient classification the
    dispatcher can route on, not just a free-text reason.
    """
    from hermes_cli import kanban_db as _kb

    board = _kb.get_current_board()
    conn = _kb.connect(board=board)
    try:
        task_id = _kb.create_task(
            conn,
            title="judge transport outage",
            body="acceptance",
            assignee="platformops",
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    threshold = goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    transport_only = ("continue", "xAI HTTP 502 via routeplane", False, None, True)
    _patch_judge_with_transport(monkeypatch, [transport_only] * (threshold + 1))

    def _block(reason: str, *, kind: "str | None" = None) -> None:
        # Mirror cli.py's _block wiring so this test exercises the same
        # production code path: forward ``kind`` into block_task so the
        # typed lifecycle is the actual persistence.
        c = _kb.connect(board=board)
        try:
            _kb.block_task(c, task_id, reason=reason, kind=kind)
        finally:
            try:
                c.close()
            except Exception:
                pass

    result = goals.run_kanban_goal_loop(
        task_id=task_id,
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=_block,
        first_response="r1",
        max_turns=20,
    )

    assert result["outcome"] == "blocked_transport"

    conn = _kb.connect(board=board)
    try:
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row is not None, f"task {task_id} missing after transport block"
        assert row["status"] == "blocked", row
        assert row["block_kind"] == "transient", (
            f"expected typed transient block_kind for transport trip, "
            f"got {row['block_kind']!r}"
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def test_block_fn_without_kind_kwarg_still_works(monkeypatch):
    """Defensive: a legacy ``block_fn(reason)`` callback (no ``kind`` kwarg)
    must keep working — the typed ``kind="transient"`` is a contract
    enhancement, not a breaking change. This protects callers that haven't
    yet been updated to accept the new kwarg.
    """
    threshold = goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    transport_only = ("continue", "xAI 502", False, None, True)
    _patch_judge_with_transport(monkeypatch, [transport_only] * (threshold + 1))

    legacy_calls = []

    def _legacy_block(reason):  # No ``kind`` parameter — must still work.
        legacy_calls.append(reason)

    result = goals.run_kanban_goal_loop(
        task_id="t_legacy_block_fn",
        goal_text="g",
        run_turn=lambda p: "x",
        task_status_fn=lambda: "running",
        block_fn=_legacy_block,
        first_response="r1",
        max_turns=20,
    )

    assert result["outcome"] == "blocked_transport"
    assert len(legacy_calls) == 1
    # Same truthful transport reason text reaches the legacy callback.
    assert "transport" in legacy_calls[0].lower()
