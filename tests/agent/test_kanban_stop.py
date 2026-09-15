"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_auto_block_after_exhaustion,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def _complete_call(call_id: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def test_nudge_after_refused_kanban_complete(clear_kanban_env):
    """A gate-refused kanban_complete leaves the card running — the guard must still fire."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_91ba131e")
    messages = _complete_call(
        "c1", '{"error": "kanban_complete refused: production_effect=production missing"}'
    )
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_no_nudge_when_retry_after_refusal_succeeds(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_91ba131e")
    messages = _complete_call("c1", '{"error": "kanban_complete refused: result too short"}')
    messages += _complete_call("c2", '{"ok": true, "task_id": "t_91ba131e"}')
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_unanswered_terminal_call_still_counts(clear_kanban_env):
    messages = _complete_call("c1", "")[:1]
    assert session_called_kanban_terminal(messages) is True






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 4 attempts after OPT-B; before OPT-B it was 2),
# and if the worker still exits without a terminal call, the auto-block
# fallback (added 2026-09-15, t_42cc3f9d) fires a synthetic kanban_block
# so the dispatcher doesn't keep re-spawning as protocol_violation.
# See also tests/hermes_cli/test_kanban_core_functionality.py for the
# dispatcher-side streak tests.



# ── OPT-B: auto-block on nudge-budget exhaustion ─────────────────────


def test_auto_block_none_when_budget_not_exhausted(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_b1")
    # attempts=1 with default max_attempts=4 → still under budget → no auto-block
    out = kanban_auto_block_after_exhaustion(messages=[], attempts=1)
    assert out is None


def test_auto_block_none_when_worker_already_terminated(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_b2")
    # attempts way past budget BUT the messages already contain a kanban_complete
    messages = _complete_call("c1", '{"ok": true}')
    out = kanban_auto_block_after_exhaustion(messages=messages, attempts=10)
    assert out is None


def test_auto_block_none_when_guard_disabled(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_b3")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    out = kanban_auto_block_after_exhaustion(messages=[], attempts=10)
    assert out is None


def test_auto_block_fires_synthetic_block_call(clear_kanban_env, monkeypatch):
    """OPT-B happy path: budget exhausted + worker ignored all nudges +
    guard enabled → a synthetic kanban_block call lands in the messages
    with the worker-task id and a transient-kind reason."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_b4")
    monkeypatch.setattr(
        "tools.kanban_tools._handle_block",
        lambda args, **kw: '{"ok": true, "synthetic": true, "task_id": "%s"}' % args["task_id"],
        raising=False,
    )
    out = kanban_auto_block_after_exhaustion(messages=[], attempts=4)
    assert out is not None
    assert out["task_id"] == "t_b4"
    assert "auto-blocked" in out["reason"] or "without a terminal kanban tool" in out["reason"]
    assert "synthetic" in out["result"]




