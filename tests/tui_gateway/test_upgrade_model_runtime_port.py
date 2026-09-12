"""Bounded regressions for the extracted Desktop model-runtime port; no providers."""
import copy
from types import SimpleNamespace

import pytest

from tui_gateway.method_ctx import rebind
from tui_gateway import model_switch


def test_once_commit_is_ephemeral_and_restores_exact_fallback_runtime():
    calls = []
    agent = SimpleNamespace(model="original", provider="fixture", api_key="fixture-key",
                            base_url="", api_mode="chat_completions",
                            runtime_capabilities={"tools": True}, _primary_runtime={"model": "original"},
                            _fallback_chain=[{"model": "backup", "provider": "fixture"}],
                            _fallback_model={"model": "backup", "provider": "fixture"},
                            _fallback_index=1, _fallback_activated=True,
                            _rate_limited_until=123, _rate_limit_backoff_count=2)
    agent.switch_model = lambda **kw: calls.append(("switch", kw))
    namespace = {"copy": copy, "logger": SimpleNamespace(warning=lambda *a: None),
                 "_load_fallback_model": lambda: [],
                 "_restart_slash_worker": lambda *a: calls.append(("restart",)),
                 "_emit_session_info": lambda *a: None}
    for name in ("_persist_live_session_runtime", "_persist_live_session_system_prompt", "_append_model_switch_marker"):
        namespace[name] = lambda *a, **kw: pytest.fail("one-turn selection wrote durable session state")
    for name in ("_snapshot_agent_model_runtime", "_set_agent_fallback_policy", "_restore_agent_model_runtime", "_commit_agent_switch"):
        namespace[name] = rebind(getattr(model_switch, name), namespace)
    snapshot = namespace["_snapshot_agent_model_runtime"](agent)
    session = {"model_override": {"model": "original", "fallback_disabled": False}}
    result = SimpleNamespace(new_model="once", target_provider="fixture", api_key="fixture-key", base_url="", api_mode="chat_completions")
    namespace["_commit_agent_switch"]("session", session, agent, result, "original", snapshot,
                                        fail_closed_pin=True, pin_session_override=True)
    assert calls[0][1]["persist_billing_route"] is False
    assert session["model_override"]["model"] == "original"
    assert agent._fallback_chain == []
    namespace["_restore_agent_model_runtime"](agent, snapshot)
    assert agent._fallback_chain == snapshot["fallback_chain"]
    assert agent._fallback_index == 1 and agent._fallback_activated
    assert agent._rate_limited_until == 123 and agent._rate_limit_backoff_count == 2
    assert calls[-1][1]["capabilities"] == {"tools": True}
