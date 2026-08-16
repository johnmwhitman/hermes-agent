from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    def __init__(self):
        self.calls = []
        self.model = "old/model"
        self.provider = "openrouter"
        self._fallback_chain = [
            {"provider": "routeplane", "model": "subs/grok"},
        ]
        self._fallback_model = self._fallback_chain[0]
        self._fallback_index = 0

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]


class _StubCLI:
    model = "old/model"
    provider = "openrouter"
    requested_provider = "openrouter"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    base_url = "https://openrouter.ai/api/v1"
    _explicit_base_url = "https://openrouter.ai/api/v1"
    api_mode = "chat_completions"
    agent: _FakeAgent | None = None
    _pending_model_switch_note = None
    _pending_one_turn_model_restore: dict | None = None
    _fallback_model = [
        {"provider": "routeplane", "model": "subs/grok"},
    ]

    def _confirm_expensive_model_switch(self, result):
        return True

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global, one_turn, custom_provs=None
    ):
        import cli as cli_mod

        return cli_mod.HermesCLI._confirm_and_apply_cli_model_switch(
            self, result, persist_global, one_turn, custom_provs
        )


def test_cli_model_once_records_restore_and_does_not_persist(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    agent = _FakeAgent()
    stub.agent = agent
    stub._snapshot_model_runtime = cli_mod.HermesCLI._snapshot_model_runtime.__get__(stub)
    printed = []

    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: printed.append(str(s)))
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not persist")))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None,
            custom_providers=None,
            with_overrides=lambda **_: SimpleNamespace(user_providers=None, custom_providers=None),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_: ModelSwitchResult(
            success=True,
            new_model="claude-sonnet-4.6",
            target_provider="anthropic",
            api_key="sk-ant",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            provider_label="Anthropic",
        ),
    )
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)

    cli_mod.HermesCLI._handle_model_switch(
        stub,
        "/model claude-sonnet-4.6 --provider anthropic --once",
    )
    cli_mod.HermesCLI._handle_model_switch(
        stub,  # type: ignore[arg-type]
        "/model claude-sonnet-4.6 --provider anthropic --once",
    )

    assert stub.model == "claude-sonnet-4.6"
    assert stub.provider == "anthropic"
    assert agent.calls[-1]["new_model"] == "claude-sonnet-4.6"
    assert agent.calls[-1]["persist_billing_route"] is False
    assert agent._fallback_chain == []
    assert stub._fallback_model == []
    restore = stub._pending_one_turn_model_restore
    assert restore is not None
    assert restore["model"] == "old/model"
    assert restore["agent_fallback_chain"] == [
        {"provider": "routeplane", "model": "subs/grok"},
    ]
    assert "next turn only" in printed[-1]


def test_cli_restore_model_runtime_snapshot_restores_agent():
    import cli as cli_mod

    stub = _StubCLI()
    agent = _FakeAgent()
    stub.agent = agent
    snapshot = {
        "model": "old/model",
        "provider": "openrouter",
        "requested_provider": "openrouter",
        "api_key": "sk-old",
        "explicit_api_key": "sk-old",
        "base_url": "https://openrouter.ai/api/v1",
        "explicit_base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "cli_fallback_model": [
            {"provider": "routeplane", "model": "subs/grok"},
        ],
        "agent_fallback_chain": [
            {"provider": "routeplane", "model": "subs/grok"},
        ],
        "agent_fallback_model": {
            "provider": "routeplane",
            "model": "subs/grok",
        },
        "agent_fallback_index": 0,
    }

    cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snapshot)

    assert stub.model == "old/model"
    assert stub.provider == "openrouter"
    assert agent.calls[-1]["new_model"] == "old/model"
    assert stub._fallback_model == [
        {"provider": "routeplane", "model": "subs/grok"},
    ]
    assert agent._fallback_chain == [
        {"provider": "routeplane", "model": "subs/grok"},
    ]


def test_cli_restore_model_runtime_preserves_active_fallback_state():
    import cli as cli_mod

    class Agent(_FakeAgent):
        _primary_runtime = None
        _rate_limited_until = 123
        _rate_limit_backoff_count = 0
        _fallback_activated = False

        def __init__(self):
            super().__init__()
            self.model = "temp/model"
            self.provider = "anthropic"

    stub = _StubCLI()
    stub.agent = Agent()
    snapshot = {
        "model": "subs/grok",
        "provider": "routeplane",
        "requested_provider": "routeplane",
        "api_key": "sk-fallback",
        "explicit_api_key": "sk-fallback",
        "base_url": "",
        "explicit_base_url": "",
        "api_mode": "chat_completions",
        "agent_primary_runtime": {
            "model": "old/model",
            "provider": "openrouter",
        },
        "agent_fallback_chain": [
            {"provider": "routeplane", "model": "subs/grok"},
        ],
        "agent_fallback_model": {
            "provider": "routeplane",
            "model": "subs/grok",
        },
        "agent_fallback_index": 0,
        "agent_fallback_activated": True,
        "agent_rate_limited_until": 123,
        "agent_rate_limit_backoff_count": 2,
    }

    cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snapshot)

    assert stub.agent.model == "subs/grok"
    assert stub.agent.provider == "routeplane"
    assert stub.agent.calls[-1]["new_model"] == "subs/grok"
    assert stub.agent._primary_runtime == {
        "model": "old/model",
        "provider": "openrouter",
    }
    assert stub.agent._fallback_activated is True
    assert stub.agent._rate_limited_until == 123
    assert stub.agent._rate_limit_backoff_count == 2


def test_cli_global_fallback_policy_rehydrates_pruned_chain(monkeypatch):
    import cli as cli_mod

    expected = [{"provider": "routeplane", "model": "subs/grok"}]
    stub = _StubCLI()
    agent = _FakeAgent()
    agent._fallback_chain = []
    stub.agent = agent
    stub._fallback_model = []
    monkeypatch.setattr(cli_mod, "get_fallback_chain", lambda _cfg: expected)

    cli_mod.HermesCLI._set_model_switch_fallback_policy(
        stub,  # type: ignore[arg-type]
        disabled=False,
    )

    assert stub._fallback_model == expected
    assert agent._fallback_chain == expected
    assert agent._fallback_model == expected[0]
    assert agent._fallback_index == 0


def test_agent_session_creation_uses_pre_once_runtime(monkeypatch):
    import run_agent

    class DB:
        created = None

        def create_session(self, **kwargs):
            self.created = kwargs

    db = DB()
    agent = SimpleNamespace(
        _persist_disabled=False,
        _session_db_created=False,
        _session_db=db,
        _session_init_model_config={"model": "temp/model", "provider": "anthropic"},
        _session_create_runtime_override={
            "model": "old/model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
        model="temp/model",
        platform="cli",
        session_id="once-session",
        _cached_system_prompt="system",
        _parent_session_id=None,
    )
    monkeypatch.setattr(run_agent, "_session_source_for_agent", lambda _platform: "cli")
    monkeypatch.setattr(run_agent, "_launch_cwd_for_session", lambda _source: "/tmp")

    run_agent.AIAgent._ensure_db_session(agent)  # type: ignore[arg-type]

    assert db.created is not None
    assert db.created["model"] == "old/model"
    assert db.created["model_config"] == {
        "model": "old/model",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }


def test_cli_arms_pre_once_runtime_before_agent_initialization():
    import cli as cli_mod

    stub = _StubCLI()
    stub._pending_one_turn_model_restore = {
        "model": "old/model",
        "provider": "openrouter",
    }
    agent = SimpleNamespace(_session_create_runtime_override=None)

    cli_mod.HermesCLI._arm_pending_one_turn_session_runtime(stub, agent)

    assert agent._session_create_runtime_override == {
        "model": "old/model",
        "provider": "openrouter",
    }


def test_cli_restore_failure_does_not_claim_runtime_restored():
    import cli as cli_mod

    class Agent(_FakeAgent):
        _primary_runtime = {"model": "temp/model", "provider": "anthropic"}
        _fallback_activated = False
        _rate_limited_until = 0
        _rate_limit_backoff_count = 0

        def switch_model(self, **_kwargs):
            raise RuntimeError("restore failed")

    stub = _StubCLI()
    stub.agent = Agent()
    snapshot = {
        "model": "old/model",
        "provider": "openrouter",
        "agent_model": "subs/grok",
        "agent_provider": "routeplane",
        "agent_primary_runtime": {"model": "old/model", "provider": "openrouter"},
        "agent_fallback_activated": True,
        "agent_rate_limited_until": 123,
        "agent_rate_limit_backoff_count": 2,
    }

    with pytest.raises(RuntimeError, match="restore failed"):
        cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snapshot)

    assert stub.agent._primary_runtime == {
        "model": "temp/model",
        "provider": "anthropic",
    }
    assert stub.agent._fallback_activated is False


def test_cli_blocked_restore_is_retained_until_retry_succeeds(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    snapshot = {"model": "old/model", "provider": "openrouter"}
    stub._blocked_one_turn_model_restore = snapshot
    monkeypatch.setattr(
        stub,
        "_restore_model_runtime_snapshot",
        lambda _snapshot: (_ for _ in ()).throw(RuntimeError("still blocked")),
        raising=False,
    )

    assert cli_mod.HermesCLI._retry_blocked_one_turn_model_restore(stub) is False
    assert stub._blocked_one_turn_model_restore is snapshot


def test_cli_one_turn_restore_restores_agent_route_signature():
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    original_signature = ("old/model", "openrouter", "openrouter", "", "", "", ())
    stub._active_agent_route_signature = original_signature
    snapshot = cli_mod.HermesCLI._snapshot_model_runtime(stub)  # type: ignore[arg-type]
    stub._active_agent_route_signature = (
        "claude-fable-5",
        "anthropic",
        "anthropic",
        "",
        "",
        "",
        (),
    )

    cli_mod.HermesCLI._restore_model_runtime_snapshot(stub, snapshot)  # type: ignore[arg-type]

    assert stub._active_agent_route_signature == original_signature
