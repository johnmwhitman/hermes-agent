"""Owner MissingSessionID repair, carried at v0.21.2's canonical request boundary.

The release already merges this header around every transport and auxiliary
request. Test that live boundary rather than duplicating old low-level headers.
"""

from __future__ import annotations

import pytest

from agent.chat_completion_helpers import build_api_kwargs
from agent.portal_tags import (
    reset_affinity_scope, reset_conversation_context,
    set_affinity_scope, set_conversation_context,
)
from run_agent import AIAgent

_MESSAGES = [{"role": "user", "content": "header fixture"}]
_ROUTES = [("kimi-k2.6", "chat_completions", "messages"),
           ("gpt-5.6-luna", "codex_responses", "input")]


@pytest.fixture(autouse=True)
def isolated_affinity():
    affinity = set_affinity_scope(None)
    conversation = set_conversation_context(None)
    try:
        yield
    finally:
        reset_conversation_context(conversation)
        reset_affinity_scope(affinity)


def _agent(model, mode, *, provider="opencode-go", base_url="https://opencode.ai/zen/go/v1"):
    agent = AIAgent(
        api_key="fixture-not-a-secret", model=model, provider=provider,
        base_url=base_url, session_id="cron_job42_20260912_120000",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[],
    )
    agent.api_mode = mode
    agent._transport = None
    return agent


@pytest.mark.parametrize("model,mode,payload_key", _ROUTES)
def test_actual_request_boundary_carries_stable_opencode_session(model, mode, payload_key):
    agent = _agent(model, mode)
    first = build_api_kwargs(agent, _MESSAGES, tools_for_api=[])
    assert payload_key in first  # exercised the actual Chat Completions / Responses builder
    # Upstream intentionally normalizes per-fire cron timestamps to one job scope.
    assert first["extra_headers"]["x-opencode-session"] == "cron_job42"
    agent.session_id = "cron_job42_20260912_130000"
    second = build_api_kwargs(agent, _MESSAGES, tools_for_api=[])
    assert second["extra_headers"]["x-opencode-session"] == "cron_job42"


@pytest.mark.parametrize("model,mode,payload_key", _ROUTES)
def test_explicit_header_survives_actual_request_boundary(model, mode, payload_key):
    agent = _agent(model, mode)
    agent.request_overrides = {"extra_headers": {
        "x-opencode-session": "caller-owned-affinity", "x-fixture": "retained",
    }}
    kwargs = build_api_kwargs(agent, _MESSAGES, tools_for_api=[])
    assert kwargs["extra_headers"]["x-opencode-session"] == "caller-owned-affinity"
    assert kwargs["extra_headers"]["x-fixture"] == "retained"


@pytest.mark.parametrize("model,mode,payload_key", _ROUTES)
def test_routeplane_custom_route_does_not_gain_opencode_header(model, mode, payload_key):
    agent = _agent(model, mode, provider="custom", base_url="http://127.0.0.1:4356/v1")
    kwargs = build_api_kwargs(agent, _MESSAGES, tools_for_api=[])
    assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})


@pytest.mark.parametrize("model,mode,payload_key", _ROUTES)
def test_missing_session_does_not_emit_empty_header(model, mode, payload_key):
    agent = _agent(model, mode)
    agent.session_id = None
    conversation = set_conversation_context(None)
    affinity = set_affinity_scope(None)
    try:
        kwargs = build_api_kwargs(agent, _MESSAGES, tools_for_api=[])
        assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})
    finally:
        reset_affinity_scope(affinity)
        reset_conversation_context(conversation)


def test_declared_host_scope_wins_over_physical_session():
    agent = _agent("gpt-5.6-luna", "codex_responses")
    conversation = set_conversation_context("conversation-root")
    affinity = set_affinity_scope("host-declared-chat")
    try:
        kwargs = build_api_kwargs(agent, _MESSAGES, tools_for_api=[])
        assert kwargs["extra_headers"]["x-opencode-session"] == "host-declared-chat"
    finally:
        reset_affinity_scope(affinity)
        reset_conversation_context(conversation)
