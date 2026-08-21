"""MCP 2.0 SEP-2322 multi-round-trip (InputRequiredResult) wrapper tests.

A 2026-07-28 server can return InputRequiredResult from tools/call (and
from resources/read / prompts/get) instead of raising a mid-RPC
elicitation. Hermes talks to ClientSession directly, so the wrapper must
opt in with allow_input_required=True and drive the retry loop through
the same elicitation callbacks that already hit the approval queue.
Without that, the SDK raises RuntimeError and cron lanes see a tool
error instead of an approval prompt.

These tests mock the transport and the approval surface — no live MCP
server and no user input. They skip cleanly if the optional `mcp` SDK
is not installed or is too old to export InputRequiredResult (the
canonical runner's .venv is still on mcp 1.x).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


pytest.importorskip("mcp.types")

import mcp.types as _mcp_types  # noqa: E402

if not hasattr(_mcp_types, "InputRequiredResult"):
    pytest.skip(
        "mcp 2.0 InputRequiredResult not available",
        allow_module_level=True,
    )

from mcp.types import (  # noqa: E402
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    ElicitResult,
    InputRequiredResult,
)

from tools import mcp_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeContentBlock:
    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeCallToolResult:
    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _form_elicit(message="authorize a payment of $0.50", schema=None):
    return ElicitRequest(
        params=ElicitRequestFormParams(
            message=message,
            requested_schema=schema
            or {
                "type": "object",
                "properties": {"approved": {"type": "boolean"}},
            },
        )
    )


def _url_elicit(message="open this url", url="https://example.com/auth"):
    return ElicitRequest(
        params=ElicitRequestURLParams(
            message=message,
            url=url,
            elicitation_id="e1",
        )
    )


def _input_required(requests, request_state="state-abc"):
    return InputRequiredResult(
        input_requests=requests,
        request_state=request_state,
    )


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:

        async def _install_lock_and_run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro

        return loop.run_until_complete(_install_lock_and_run())
    finally:
        loop.close()


class _Session2:
    """ClientSession-shaped surface: named allow_input_required kwarg.

    Production inspects the live method signature before opting in, so a
    bare AsyncMock (*args, **kwargs) would not prove the flag is passed
    and would also break older exact-args assertions. This recorder
    mirrors the mcp 2.0 ClientSession.call_tool / read_resource /
    get_prompt signatures.
    """

    def __init__(self):
        self.call_tool_calls = []
        self.read_resource_calls = []
        self.get_prompt_calls = []
        self.call_tool_impl = None
        self.read_resource_impl = None
        self.get_prompt_impl = None

    async def call_tool(
        self,
        name,
        arguments=None,
        *,
        allow_input_required=False,
        input_responses=None,
        request_state=None,
        meta=None,
        allow_claimed=False,
    ):
        self.call_tool_calls.append(
            {
                "name": name,
                "arguments": arguments,
                "allow_input_required": allow_input_required,
                "input_responses": input_responses,
                "request_state": request_state,
            }
        )
        if self.call_tool_impl is not None:
            return await self.call_tool_impl(
                name,
                arguments,
                allow_input_required=allow_input_required,
                input_responses=input_responses,
                request_state=request_state,
            )
        return _FakeCallToolResult(content=[_FakeContentBlock("ok")])

    async def read_resource(
        self,
        uri,
        *,
        allow_input_required=False,
        input_responses=None,
        request_state=None,
        meta=None,
    ):
        self.read_resource_calls.append(
            {
                "uri": uri,
                "allow_input_required": allow_input_required,
            }
        )
        if self.read_resource_impl is not None:
            return await self.read_resource_impl(uri)
        return SimpleNamespace(
            contents=[SimpleNamespace(text="file body", blob=None)]
        )

    async def get_prompt(
        self,
        name,
        arguments=None,
        *,
        allow_input_required=False,
        input_responses=None,
        request_state=None,
        meta=None,
    ):
        self.get_prompt_calls.append(
            {
                "name": name,
                "arguments": arguments,
                "allow_input_required": allow_input_required,
            }
        )
        if self.get_prompt_impl is not None:
            return await self.get_prompt_impl(name, arguments)
        return SimpleNamespace(
            messages=[
                SimpleNamespace(
                    role="assistant",
                    content=SimpleNamespace(text="a summary"),
                )
            ],
            description=None,
        )


@pytest.fixture
def connected_server():
    """A fake connected MCP server + loop; yield the server namespace."""
    session = _Session2()
    server = SimpleNamespace(
        name="pay",
        session=session,
        _rpc_lock=None,
        _elicitation=mcp_tool.ElicitationHandler("pay", {"timeout": 5}),
        _sampling=None,
        _pending_call_context=None,
    )
    server._elicitation.owner = server
    with patch.dict(mcp_tool._servers, {"pay": server}), patch(
        "tools.mcp_tool._run_on_mcp_loop",
        side_effect=_fake_run_on_mcp_loop,
    ), patch.dict(mcp_tool._server_error_counts, {}, clear=True), patch.dict(
        mcp_tool._server_breaker_opened_at, {}, clear=True
    ), patch.dict(mcp_tool._server_trust_levels, {}, clear=True), patch.dict(
        mcp_tool._tool_read_only_hints, {}, clear=True
    ):
        yield server


# ---------------------------------------------------------------------------
# Opt-in: the first tools/call must pass allow_input_required=True
# ---------------------------------------------------------------------------


class TestCallToolOptsIntoInputRequired:
    def test_first_call_passes_allow_input_required(self, connected_server):
        """Without this flag, mcp 2.0 raises RuntimeError on InputRequiredResult."""
        handler = mcp_tool._make_tool_handler("pay", "charge", 30.0)
        raw = handler({"amount": "0.50"})
        assert json.loads(raw) == {"result": "ok"}
        assert connected_server.session.call_tool_calls, "call_tool was not invoked"
        first = connected_server.session.call_tool_calls[0]
        assert first["allow_input_required"] is True
        assert first["name"] == "charge"
        assert first["arguments"] == {"amount": "0.50"}

    def test_happy_path_still_returns_tool_text(self, connected_server):
        handler = mcp_tool._make_tool_handler("pay", "charge", 30.0)
        raw = handler({"amount": "0.50"})
        assert json.loads(raw)["result"] == "ok"


# ---------------------------------------------------------------------------
# Form elicitation: InputRequiredResult must hit the approval queue
# ---------------------------------------------------------------------------


class TestFormElicitationReachesApprovalQueue:
    def test_accept_retries_with_input_responses_and_request_state(
        self, connected_server
    ):
        elicit = _form_elicit()
        first = _input_required({"k1": elicit}, request_state="state-abc")
        terminal = _FakeCallToolResult(content=[_FakeContentBlock("charged")])

        async def _impl(
            name,
            arguments=None,
            *,
            allow_input_required=False,
            input_responses=None,
            request_state=None,
        ):
            if input_responses is None:
                return first
            return terminal

        connected_server.session.call_tool_impl = _impl

        handler = mcp_tool._make_tool_handler("pay", "charge", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
            return_value="accept",
        ) as consent:
            raw = handler({"amount": "0.50"})

        consent.assert_called_once()
        assert "authorize a payment" in consent.call_args.args[0]
        parsed = json.loads(raw)
        assert parsed["result"] == "charged"
        assert "error" not in parsed

        calls = connected_server.session.call_tool_calls
        assert len(calls) == 2
        assert calls[0]["allow_input_required"] is True
        assert calls[1]["allow_input_required"] is True
        assert calls[1]["request_state"] == "state-abc"
        responses = calls[1]["input_responses"] or {}
        assert "k1" in responses
        assert isinstance(responses["k1"], ElicitResult)
        assert responses["k1"].action == "accept"

    def test_decline_does_not_retry_as_success(self, connected_server):
        elicit = _form_elicit()
        first = _input_required({"k1": elicit}, request_state="state-abc")

        async def _impl(
            name,
            arguments=None,
            *,
            allow_input_required=False,
            input_responses=None,
            request_state=None,
        ):
            return first

        connected_server.session.call_tool_impl = _impl

        handler = mcp_tool._make_tool_handler("pay", "charge", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
            return_value="decline",
        ) as consent:
            raw = handler({"amount": "0.50"})

        consent.assert_called_once()
        parsed = json.loads(raw)
        assert "error" in parsed
        assert "charged" not in parsed.get("result", "")
        # Fail-closed: decline becomes ErrorData and the driver aborts
        # without a success retry.
        retry_calls = [
            c
            for c in connected_server.session.call_tool_calls
            if c.get("input_responses")
        ]
        assert retry_calls == []

    def test_url_mode_is_declined_without_opening_a_browser(
        self, connected_server
    ):
        first = _input_required({"k1": _url_elicit()}, request_state="url-state")

        async def _impl(
            name,
            arguments=None,
            *,
            allow_input_required=False,
            input_responses=None,
            request_state=None,
        ):
            return first

        connected_server.session.call_tool_impl = _impl

        handler = mcp_tool._make_tool_handler("pay", "charge", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
        ) as consent:
            raw = handler({"amount": "0.50"})

        consent.assert_not_called()
        parsed = json.loads(raw)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# Sampling InputRequiredResult must not grow the sampling surface
# ---------------------------------------------------------------------------


class TestSamplingInputRequiredDoesNotGrowSurface:
    def test_sampling_request_is_refused_and_does_not_call_llm(
        self, connected_server
    ):
        from mcp.types import (
            CreateMessageRequest,
            CreateMessageRequestParams,
            SamplingMessage,
            TextContent,
        )

        sample = CreateMessageRequest(
            params=CreateMessageRequestParams(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text="hi"),
                    )
                ],
                max_tokens=16,
            )
        )
        first = _input_required({"s1": sample}, request_state="sample-state")

        async def _impl(
            name,
            arguments=None,
            *,
            allow_input_required=False,
            input_responses=None,
            request_state=None,
        ):
            return first

        connected_server.session.call_tool_impl = _impl
        connected_server._sampling = SimpleNamespace(
            __call__=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("sampling must not grow")
            )
        )

        handler = mcp_tool._make_tool_handler("pay", "charge", 30.0)
        with patch.object(
            mcp_tool.SamplingHandler,
            "__call__",
            side_effect=AssertionError("sampling must not grow"),
        ):
            raw = handler({"amount": "0.50"})

        parsed = json.loads(raw)
        assert "error" in parsed
        retry_calls = [
            c
            for c in connected_server.session.call_tool_calls
            if c.get("input_responses")
        ]
        assert retry_calls == []


# ---------------------------------------------------------------------------
# Utility methods that also gained allow_input_required in mcp 2.0
# ---------------------------------------------------------------------------


class TestUtilityMethodsOptIn:
    def test_read_resource_passes_allow_input_required(self, connected_server):
        handler = mcp_tool._make_read_resource_handler("pay", 30.0)
        raw = handler({"uri": "file:///tmp/x"})
        assert "file body" in json.loads(raw)["result"]
        assert connected_server.session.read_resource_calls
        assert (
            connected_server.session.read_resource_calls[0]["allow_input_required"]
            is True
        )

    def test_get_prompt_passes_allow_input_required(self, connected_server):
        handler = mcp_tool._make_get_prompt_handler("pay", 30.0)
        raw = handler({"name": "summarize", "arguments": {"text": "hi"}})
        assert json.loads(raw)["messages"][0]["content"] == "a summary"
        assert connected_server.session.get_prompt_calls
        assert (
            connected_server.session.get_prompt_calls[0]["allow_input_required"]
            is True
        )
