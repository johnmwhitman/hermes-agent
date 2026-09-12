"""Bounded MCP 2.0 InputRequired driver; form approval only, never sampling or URL consent."""

import importlib
import inspect
import logging
from types import SimpleNamespace
from typing import Any
from tools.mcp_tool_sampling import ElicitationHandler

logger = logging.getLogger("tools.mcp_tool")

# ---------------------------------------------------------------------------
# MCP 2.0 SEP-2322 multi-round-trip (InputRequiredResult)
# ---------------------------------------------------------------------------
#
# Hermes talks to ClientSession directly (not mcp.client.Client). On mcp
# 2.0, tools/call / resources/read / prompts/get can return
# InputRequiredResult instead of raising a mid-RPC elicitation. The SDK
# raises RuntimeError unless the caller passes allow_input_required=True
# and drives the retry loop. Form elicitation is routed through the
# existing approval queue; URL-mode and sampling requests are refused
# so this patch does not grow the sampling surface.

_INPUT_REQUIRED_MAX_ROUNDS = 10


def _input_required_type():
    """Return the live InputRequiredResult class, or None on mcp < 2.0."""
    try:
        types = importlib.import_module("mcp.types")
    except Exception:
        return None
    return getattr(types, "InputRequiredResult", None)


def _session_accepts_kwarg(method: Any, name: str) -> bool:
    """True when *method* explicitly names *name* (not just ``**kwargs``).

    AsyncMock / MagicMock advertise ``(*args, **kwargs)``, which would
    otherwise make every mocked 1.x session look 2.0-capable and break
    existing ``assert_called_once_with(name, arguments=...)`` tests.
    """
    try:
        param = inspect.signature(method).parameters.get(name)
    except (TypeError, ValueError):
        return False
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _input_required_call_kwargs(method: Any, **extra: Any) -> dict:
    """Kwargs to opt a ClientSession method into InputRequiredResult.

    mcp 1.x ClientSession.call_tool has no ``allow_input_required``
    parameter and would TypeError if we always passed it. Probe the
    live signature so 1.x stays byte-identical and 2.0 opts in.
    """
    kwargs = dict(extra)
    if _session_accepts_kwarg(method, "allow_input_required"):
        kwargs["allow_input_required"] = True
    return kwargs


def _is_input_required_result(result: Any) -> bool:
    cls = _input_required_type()
    return cls is not None and isinstance(result, cls)


def _input_request_kind(request: Any) -> str:
    """Classify an embedded InputRequest as elicit / sampling / roots / other."""
    method = getattr(request, "method", "") or ""
    if method.startswith("elicitation/"):
        return "elicit"
    if method.startswith("sampling/"):
        return "sampling"
    if "root" in method:
        return "roots"
    name = type(request).__name__
    if "Elicit" in name:
        return "elicit"
    if "CreateMessage" in name or "Sampling" in name:
        return "sampling"
    if "Root" in name:
        return "roots"
    return "other"


def _error_data(message: str, code: int = -32600) -> Any:
    """Build an SDK ErrorData if available, else a decline-shaped stand-in."""
    try:
        types = importlib.import_module("mcp.types")
        cls = getattr(types, "ErrorData", None)
        if cls is not None:
            return cls(code=code, message=message)
    except Exception:
        pass
    return SimpleNamespace(code=code, message=message)


async def _dispatch_input_request(server: Any, key: str, request: Any) -> Any:
    """Fulfil one embedded InputRequest, or refuse it fail-closed.

    Form elicitation reuses ElicitationHandler so the approval queue is
    the only user-facing surface. URL-mode elicitation is already
    declined by that handler. Sampling and roots are refused here so
    this wiring cannot grow those surfaces.
    """
    kind = _input_request_kind(request)
    if kind != "elicit":
        logger.info(
            "MCP server '%s' InputRequired '%s' request refused (%s)",
            getattr(server, "name", "?"), key, kind,
        )
        return _error_data(
            f"Hermes does not fulfil MCP {kind} InputRequired requests "
            f"from this wrapper; form elicitation only."
        )

    handler = getattr(server, "_elicitation", None)
    if handler is None:
        # Session may have been constructed without an elicitation
        # callback (config disabled it, or SDK types were missing at
        # connect). Build an ephemeral handler so a 2.0 server that
        # still returned InputRequiredResult can reach the approval
        # queue instead of raising.
        handler = ElicitationHandler(
            getattr(server, "name", "mcp"), {},
            call_context=lambda: getattr(server, "_pending_call_context", None),
        )

    params = getattr(request, "params", None)
    if params is None:
        return _error_data("InputRequired elicitation request had no params")

    # Drive the existing callback. On a live ClientSession the SDK's
    # dispatch_input_request does the same thing; we call the handler
    # directly so mocked sessions (and sessions whose callback table
    # was never wired) still hit the approval queue.
    result = await handler(context=None, params=params)
    action = getattr(result, "action", None)
    if action == "accept":
        return result
    return _error_data(
        f"User {action or 'declined'} the MCP elicitation request"
    )


async def _drive_input_required(
    server: Any,
    first: Any,
    retry: Any,
    *,
    max_rounds: int = _INPUT_REQUIRED_MAX_ROUNDS,
) -> Any:
    """Resolve an InputRequiredResult via the SDK driver, or a local loop.

    Prefer ``mcp.client._input_required.run_input_required_driver`` so
    we inherit the official max-rounds / state-only backoff behaviour.
    Fall back to a small local loop if that helper is missing.
    """
    if not _is_input_required_result(first):
        return first

    async def dispatch(key: str, req: Any) -> Any:
        return await _dispatch_input_request(server, key, req)

    try:
        from mcp.client._input_required import run_input_required_driver
    except ImportError:
        run_input_required_driver = None  # type: ignore[assignment]

    if run_input_required_driver is not None:
        return await run_input_required_driver(
            first, dispatch=dispatch, retry=retry, max_rounds=max_rounds,
        )

    # Local fallback for SDKs that expose InputRequiredResult but not
    # the driver module (should not happen on mcp 2.0, but keep the
    # wrapper useful if the helper moves).
    current = first
    for _ in range(max_rounds):
        if not _is_input_required_result(current):
            return current
        requests = getattr(current, "input_requests", None) or {}
        state = getattr(current, "request_state", None)
        responses: dict = {}
        for key, req in requests.items():
            result = await dispatch(key, req)
            if type(result).__name__ == "ErrorData" or getattr(result, "code", None) is not None:
                raise RuntimeError(getattr(result, "message", "input request refused"))
            responses[key] = result
        current = await retry(responses or None, state)
    raise RuntimeError(
        f"Server returned InputRequiredResult for more than {max_rounds} rounds"
    )


async def _call_with_input_required(
    server: Any,
    method: Any,
    *args: Any,
    **base_kwargs: Any,
) -> Any:
    """Invoke a ClientSession method, driving InputRequiredResult if returned."""
    call_kwargs = _input_required_call_kwargs(method, **base_kwargs)
    first = await method(*args, **call_kwargs)
    if not _is_input_required_result(first):
        return first

    async def retry(responses, state):
        retry_kwargs = dict(call_kwargs)
        if _session_accepts_kwarg(method, "input_responses"):
            retry_kwargs["input_responses"] = responses
        if _session_accepts_kwarg(method, "request_state"):
            retry_kwargs["request_state"] = state
        return await method(*args, **retry_kwargs)

    return await _drive_input_required(server, first, retry)
