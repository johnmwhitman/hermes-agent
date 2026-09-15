"""Turn-end guard for kanban workers, which must end with ``kanban_complete`` or
``kanban_block``. Some models narrate the next step and stop with no tool calls;
Hermes treats that as a clean exit → ``rc=0`` → dispatcher ``protocol_violation``.
Policy-only: return a bounded synthetic nudge so the loop continues instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

_DEFAULT_MAX_ATTEMPTS = 4  # Some model families (GLM/Qwen) ignore early nudges — let them reach budget.


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def _tool_call_id(tc: Any) -> str:
    return str((tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")) or "")


def _is_tool_error(content: Any) -> bool:
    """A ``tool_error`` body (``{"error": ...}``) — e.g. ``kanban_complete refused: ...``."""
    if not isinstance(content, str):
        return False
    try:
        body = json.loads(content)
    except (ValueError, TypeError):
        return False
    return isinstance(body, dict) and "error" in body


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already made a terminal kanban call that was not refused.

    A refused call (gate rejection, ``tool_error`` result) leaves the card ``running``;
    counting it would silence the nudge exactly when the worker still has to retry.
    A terminal call with no result row yet counts (it may still land).
    """
    pending: set[str] = set()
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    call_id = _tool_call_id(tc)
                    if not call_id:
                        return True  # cannot pair with its result — legacy behaviour
                    pending.add(call_id)
        elif role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            if call_id in pending:
                pending.discard(call_id)
                if not _is_tool_error(msg.get("content")):
                    return True
            elif not call_id and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
                if not _is_tool_error(msg.get("content")):
                    return True
    return bool(pending)


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal", "kanban_auto_block_after_exhaustion"]


def kanban_auto_block_after_exhaustion(messages, attempts, max_attempts=_DEFAULT_MAX_ATTEMPTS):
    """OPT-B: when the nudge budget is exhausted AND the worker still hasn't called
    kanban_complete / kanban_block, synthesize a `kanban_block` tool invocation so the
    conversation ends with a terminal record — the dispatcher sees a real terminal call,
    not a `protocol_violation` retry.

    Returns a dict with ``call_id``, ``reason``, ``result`` (handler return text), or ``None``
    when the budget is not yet exhausted / the worker already terminated / the guard is
    disabled. The caller (turn_stop_gates) is responsible for appending the synthetic
    assistant + tool rows and continuing with ``continue_turn=False``.
    """
    if (
        not kanban_stop_nudge_enabled()
        or attempts < max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    reason = (
        f"worker exited without a terminal kanban tool after "
        f"{attempts} nudge attempts (guard budget={max_attempts}); "
        f"auto-blocked to prevent dispatcher protocol_violation re-spawn loop. "
        f"Human review required."
    )
    try:
        from tools.kanban_tools import _handle_block
        handler_result = _handle_block({"task_id": tid, "reason": reason, "kind": "transient"})
    except Exception as exc:
        # If the handler refuses (e.g. ownership guard, missing DB) we let the
        # worker exit anyway — the dispatcher will then retry at most a few more
        # times before the per-task streak cap kicks in.
        handler_result = f"auto-block failed: {type(exc).__name__}: {exc}"

    return {"task_id": tid, "reason": reason, "result": handler_result}
