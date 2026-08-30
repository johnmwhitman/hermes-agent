from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from enum import Enum
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import posture


def test_gateway_thread_metadata_tolerates_platform_enum_without_a2a(monkeypatch):
    """Narrow gateway test doubles must not crash on an absent A2A member."""
    from gateway import run as gateway_run

    class MinimalPlatform(Enum):
        SLACK = "slack"

    monkeypatch.setattr(gateway_run, "Platform", MinimalPlatform)
    runner = object.__new__(gateway_run.GatewayRunner)
    source = SimpleNamespace(
        platform=MinimalPlatform.SLACK,
        chat_id="C_TEST",
        thread_id=None,
        chat_type="channel",
        message_id="m1",
        scope_id="T_TEST",
        user_id="U_TEST",
    )

    metadata = runner._thread_metadata_for_source(source)

    assert metadata == {
        "slack_team_id": "T_TEST",
        "scope_id": "T_TEST",
        "user_id": "U_TEST",
    }


def test_gateway_thread_metadata_normalizes_raw_a2a_platform_case():
    from gateway import run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    source = SimpleNamespace(
        platform=" A2A ",
        chat_id="ctx-1",
        thread_id=None,
        chat_type="dm",
        message_id="m1",
        a2a_peer="conductor",
        a2a_agent_slug="meshfleet",
        a2a_context_id="ctx-1",
    )

    metadata = runner._thread_metadata_for_source(source)

    assert metadata == {
        "a2a_peer": "conductor",
        "a2a_agent_slug": "meshfleet",
        "a2a_context_id": "ctx-1",
    }


@pytest.mark.asyncio
async def test_gateway_normalizes_raw_a2a_platform_before_ingress_side_effects():
    from gateway import run as gateway_run
    from gateway.platforms.base import MessageEvent

    runner = object.__new__(gateway_run.GatewayRunner)
    calls = []
    runner._validate_a2a_ingress_source = lambda source: calls.append(source) or False
    source = SimpleNamespace(platform="A2A")

    result = await gateway_run.GatewayRunner._handle_message(
        runner,
        MessageEvent(text="must be rejected", source=source),
    )

    assert result is None
    assert calls == [source]


def test_apply_to_agent_normalizes_raw_a2a_platform_case_and_narrows_tools():
    binding = posture.make_binding(
        "alice", "research", "ctx-1", ["read_file"], mutation_enabled=False,
    )
    source = SimpleNamespace(
        platform=" A2A ",
        chat_id="ctx-1",
        a2a_peer="alice",
        a2a_agent_slug="research",
        a2a_context_id="ctx-1",
        a2a_mutation_requested=False,
        a2a_mutation_enabled=False,
        a2a_credential_authenticated=True,
        a2a_peer_trusted=True,
        a2a_mutable_toolsets=(),
        a2a_allowed_tool_names=("read_file",),
        a2a_toolset_fingerprint=binding["toolset_fingerprint"],
        a2a_binding=binding,
    )
    agent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
        valid_tool_names={"read_file", "terminal"},
    )

    allowed = posture.apply_to_agent(agent, source)

    assert allowed == frozenset({"read_file"})
    assert agent.valid_tool_names == {"read_file"}
    assert agent._a2a_posture_binding_matches is True


def test_session_source_serialization_normalizes_raw_a2a_and_keeps_binding():
    from gateway.session import SessionSource

    binding = posture.make_binding(
        "alice", "research", "ctx-1", ["read_file"], mutation_enabled=False,
    )
    source = SessionSource(
        platform=" A2A ",
        chat_id="ctx-1",
        a2a_peer="alice",
        a2a_agent_slug="research",
        a2a_context_id="ctx-1",
        a2a_mutation_requested=False,
        a2a_mutation_enabled=False,
        a2a_credential_authenticated=True,
        a2a_peer_trusted=True,
        a2a_allowed_tool_names=("read_file",),
        a2a_toolset_fingerprint=binding["toolset_fingerprint"],
        a2a_binding=binding,
    )

    payload = source.to_dict()
    restored = SessionSource.from_dict(payload)

    assert payload["platform"] == "a2a"
    assert payload["a2a_binding"] == binding
    assert restored.a2a_binding == binding
    assert restored.a2a_allowed_tool_names == ("read_file",)


def test_missing_mutation_metadata_is_read_only():
    decision = posture.parse_mutation_request({"message": {"parts": []}})
    assert decision.requested is False
    assert decision.error is None


def test_mutation_metadata_requires_a_real_json_boolean():
    for value in ("true", 1, 0, None, [], {}):
        decision = posture.parse_mutation_request(
            {"message": {"metadata": {posture.MUTATION_METADATA_KEY: value}}}
        )
        assert decision.error


def test_conflicting_message_and_params_metadata_is_rejected():
    decision = posture.parse_mutation_request(
        {
            "metadata": {posture.MUTATION_METADATA_KEY: False},
            "message": {
                "metadata": {posture.MUTATION_METADATA_KEY: True},
                "parts": [],
            },
        }
    )
    assert decision.error


def test_mutable_request_needs_credential_and_peer_allowlist():
    assert not posture.effective_mutation(
        requested=True,
        credential_authenticated=False,
        trusted_peer=True,
        mutable_toolsets=["terminal"],
    )
    assert not posture.effective_mutation(
        requested=True,
        credential_authenticated=True,
        trusted_peer=False,
        mutable_toolsets=["terminal"],
    )
    assert not posture.effective_mutation(
        requested=True,
        credential_authenticated=True,
        trusted_peer=True,
        mutable_toolsets=[],
    )
    assert posture.effective_mutation(
        requested=True,
        credential_authenticated=True,
        trusted_peer=True,
        mutable_toolsets=["terminal"],
    )
    assert not posture.effective_mutation(
        requested=1,
        credential_authenticated=True,
        trusted_peer=True,
        mutable_toolsets=["terminal"],
    )
    assert not posture.effective_mutation(
        requested=True,
        credential_authenticated="yes",
        trusted_peer=True,
        mutable_toolsets=["terminal"],
    )


def test_read_only_allowlist_is_exact_and_immutable():
    assert posture.READONLY_TOOL_NAMES == frozenset(
        {
            "read_file",
            "search_files",
            "skills_list",
            "skill_view",
            "web_search",
            "web_extract",
            "kanban_show",
            "kanban_list",
            "a2a_history",
            "a2a_list",
        }
    )
    assert posture.allowed_tool_names(False, ["terminal"]) == posture.READONLY_TOOL_NAMES


def test_mutable_allowlist_excludes_unbounded_composite_tools():
    allowed = posture.allowed_tool_names(
        True, ["terminal", "execute_code", "delegate_task"],
    )

    assert "terminal" in allowed
    assert "execute_code" not in allowed
    assert "delegate_task" not in allowed


def test_resume_binding_mismatch_rejects_but_corrupt_binding_is_read_only():
    expected = posture.make_binding("alice", "research", "ctx-1", ["read_file"])
    assert posture.resume_binding_status(expected, expected) == "ok"
    other = posture.make_binding("alice", "research", "ctx-1", ["terminal"])
    assert posture.resume_binding_status(expected, other) == "mismatch"
    assert posture.resume_binding_status({"peer": "alice"}, expected) == "corrupt"


def test_binding_store_is_atomic_and_round_trips(monkeypatch, tmp_path):
    path = tmp_path / "bindings.json"
    monkeypatch.setattr(posture, "_binding_store_path", lambda: path)
    binding = posture.make_binding("alice", "research", "ctx-1", ["read_file"])
    posture.persist_bindings({("research", "ctx-1"): binding})
    loaded = posture.load_persisted_bindings()
    assert loaded[("research", "ctx-1")] == binding


def test_forwarded_child_policy_is_consumed_before_tool_dispatch(monkeypatch, tmp_path):
    binding = posture.make_binding("alice", "research", "ctx-1", ["read_file"])
    unsigned_policy = {
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-1",
        "mutation_enabled": False,
        "allowed_tool_names": ["read_file"],
        "binding": binding,
    }
    secret = b"a" * 32
    key_dir = tmp_path / "hermes-home"
    key_dir.mkdir(mode=0o700)
    key_path = key_dir / "a2a_child_issuer.key"
    key_path.write_bytes(secret)
    key_path.chmod(0o600)
    policy = json.dumps(posture.sign_child_policy(unsigned_policy, secret=secret))
    env = os.environ.copy()
    env["HERMES_HOME"] = str(key_dir)
    env[posture.CHILD_POLICY_ENV] = policy
    script = """
from plugins.platforms.a2a import posture
from model_tools import handle_function_call
p = posture.load_child_policy()
assert p and not p.get('error')
for n in ('terminal', 'write_file', 'delegate_task'):
    r = handle_function_call(n, {}, a2a_posture=p)
    assert 'A2A mutation posture' in str(r)
r = handle_function_call('read_file', {'path': '/does/not/exist'}, a2a_posture=p)
assert 'A2A mutation posture' not in str(r)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=os.getcwd(), env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_forwarded_readonly_loopback_has_zero_protected_delta(monkeypatch, tmp_path):
    protected = tmp_path / "protected.txt"
    protected.write_text("sentinel", encoding="utf-8")
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"durable-state")
    binding = posture.make_binding("alice", "research", "ctx-zero", ["read_file"])
    secret = b"z" * 32
    key_dir = tmp_path / "hermes-home"
    key_dir.mkdir(mode=0o700)
    key_path = key_dir / "a2a_child_issuer.key"
    key_path.write_bytes(secret)
    key_path.chmod(0o600)
    policy = json.dumps(posture.sign_child_policy({
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-zero",
        "mutation_enabled": False,
        "allowed_tool_names": ["read_file"],
        "binding": binding,
    }, secret=secret))
    env = os.environ.copy()
    env["HERMES_HOME"] = str(key_dir)
    env[posture.CHILD_POLICY_ENV] = policy
    script = """
import hashlib, os, psutil, sys
from plugins.platforms.a2a import posture
from model_tools import handle_function_call

protected, state_db = sys.argv[1:3]
def digest(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

before = (digest(protected), digest(state_db), {p.pid for p in psutil.Process().children(recursive=True)})
p = posture.load_child_policy()
assert p and not p.get('error')
assert 'sentinel' in str(handle_function_call('read_file', {'path': protected}, a2a_posture=p))
for name, args in (
    ('write_file', {'path': protected, 'content': 'changed'}),
    ('terminal', {'command': 'touch ' + protected}),
):
    result = handle_function_call(name, args, a2a_posture=p)
    assert 'A2A mutation posture' in str(result)
after = (digest(protected), digest(state_db), {p.pid for p in psutil.Process().children(recursive=True)})
assert before == after
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(protected), str(state_db)],
        cwd=os.getcwd(), env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_audit_redacts_before_truncating():
    raw = "x" * 450 + " sk-abcdefghij1234567890XYZ"
    redacted = posture.redact_audit_summary(raw, limit=500)
    assert "sk-abcdefghij" not in redacted
    assert "[redacted]" in redacted
    assert len(redacted) <= 500


def test_registry_backstop_denies_before_handler_lookup(monkeypatch):
    from tools.registry import registry

    def should_not_lookup(*_args, **_kwargs):
        raise AssertionError("A2A denial must precede registry lookup")

    monkeypatch.setattr(registry, "get_entry", should_not_lookup)
    result = registry.dispatch(
        "terminal", {}, a2a_allowed_tool_names=posture.READONLY_TOOL_NAMES
    )
    assert "A2A mutation posture" in str(result)


def test_forwarded_policy_rejects_decision_and_readonly_surface_mismatch():
    secret = b"b" * 32
    assert posture.load_child_policy(
        {"authenticated": True}, secret=secret,
    ).get("error")
    mutable_binding = posture.make_binding(
        "alice", "research", "ctx-1", ["terminal"], mutation_enabled=False,
    )
    bad_surface = {
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-1",
        "mutation_enabled": False,
        "allowed_tool_names": ["terminal"],
        "binding": mutable_binding,
    }
    signed_bad_surface = posture.sign_child_policy(bad_surface, secret=secret)
    assert posture.load_child_policy(signed_bad_surface, secret=secret).get("error")

    bad_decision = dict(bad_surface)
    bad_decision["mutation_enabled"] = True
    signed_bad_decision = posture.sign_child_policy(bad_decision, secret=secret)
    assert posture.load_child_policy(signed_bad_decision, secret=secret).get("error")

    valid_binding = posture.make_binding(
        "alice", "research", "ctx-1", ["read_file"], mutation_enabled=False,
    )
    valid_policy = {
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-1",
        "mutation_enabled": False,
        "allowed_tool_names": ["read_file"],
        "binding": valid_binding,
    }
    signed_valid = posture.sign_child_policy(valid_policy, secret=secret)
    normalized = posture.load_child_policy(signed_valid, secret=secret)
    assert posture.child_policy_matches_tools(normalized, ["read_file"])
    assert not posture.child_policy_matches_tools(normalized, [])


@pytest.mark.parametrize("forbidden_name", ["execute_code", "delegate_task"])
def test_forwarded_policy_rejects_unbounded_composite_tools(forbidden_name):
    secret = b"c" * 32
    binding = posture.make_binding(
        "alice", "research", "ctx-1", [forbidden_name], mutation_enabled=True,
    )
    signed = posture.sign_child_policy(
        {
            "authenticated": True,
            "served_agent_slug": "research",
            "context_id": "ctx-1",
            "mutation_enabled": True,
            "allowed_tool_names": [forbidden_name],
            "binding": binding,
        },
        secret=secret,
    )

    assert posture.load_child_policy(signed, secret=secret).get("error")


@pytest.mark.parametrize("forbidden_name", ["execute_code", "delegate_task"])
def test_direct_dispatch_rejects_unbounded_composite_tools(forbidden_name):
    from model_tools import handle_function_call
    from tools.registry import registry

    allowed = posture.allowed_tool_names(True, [forbidden_name])
    policy = {"allowed_tool_names": allowed, "binding": {}}

    direct = handle_function_call(forbidden_name, {}, a2a_posture=policy)
    registry_result = registry.dispatch(
        forbidden_name, {}, a2a_allowed_tool_names=allowed,
    )

    assert "A2A mutation posture" in str(direct)
    assert "A2A mutation posture" in str(registry_result)


@pytest.mark.parametrize("forbidden_name", ["execute_code", "delegate_task"])
def test_direct_dispatch_rejects_forged_composite_allowlist(forbidden_name):
    from model_tools import handle_function_call
    from tools.registry import registry

    forged_allowed = frozenset({forbidden_name})
    policy = {"allowed_tool_names": forged_allowed, "binding": {}}

    direct = handle_function_call(forbidden_name, {}, a2a_posture=policy)
    registry_result = registry.dispatch(
        forbidden_name, {}, a2a_allowed_tool_names=forged_allowed,
    )

    assert "A2A mutation posture" in str(direct)
    assert "A2A mutation posture" in str(registry_result)


def test_empty_direct_posture_denies_before_dispatch():
    from model_tools import handle_function_call

    result = handle_function_call("definitely_not_a_tool", {}, a2a_posture={})
    assert "A2A mutation posture" in str(result)


def test_source_binding_requires_exact_transport_fields():
    binding = posture.make_binding("alice", "research", "ctx-1", ["read_file"])
    source = SimpleNamespace(
        platform="a2a",
        chat_id="ctx-1",
        a2a_peer="alice",
        a2a_agent_slug="research",
        a2a_context_id="ctx-1",
        a2a_mutation_requested=False,
        a2a_mutation_enabled=False,
        a2a_credential_authenticated=True,
        a2a_peer_trusted=True,
        a2a_toolset_fingerprint=binding["toolset_fingerprint"],
        a2a_allowed_tool_names=("read_file",),
        a2a_binding=binding,
    )
    assert posture.source_binding_valid(source)
    source.a2a_peer = "mallory"
    assert not posture.source_binding_valid(source)
    source.a2a_peer = "alice"
    source.a2a_mutation_enabled = True
    assert not posture.source_binding_valid(source)
    source.a2a_mutation_enabled = False
    source.a2a_allowed_tool_names = ("read_file", "terminal")
    assert not posture.source_binding_valid(source)


@pytest.mark.parametrize("forbidden_name", ["execute_code", "delegate_task"])
def test_source_binding_rejects_unbounded_composite_tools(forbidden_name):
    binding = posture.make_binding(
        "alice", "research", "ctx-1", [forbidden_name], mutation_enabled=True,
    )
    source = SimpleNamespace(
        platform="a2a",
        chat_id="ctx-1",
        a2a_peer="alice",
        a2a_agent_slug="research",
        a2a_context_id="ctx-1",
        a2a_mutation_requested=True,
        a2a_mutation_enabled=True,
        a2a_credential_authenticated=True,
        a2a_peer_trusted=True,
        a2a_toolset_fingerprint=binding["toolset_fingerprint"],
        a2a_allowed_tool_names=(forbidden_name,),
        a2a_binding=binding,
    )

    assert not posture.source_binding_valid(source)


def test_only_a_live_adapter_can_stamp_a_valid_ingress_source():
    from gateway.config import PlatformConfig
    from gateway.session import SessionSource
    from plugins.platforms.a2a.adapter import A2AAdapter

    adapter = A2AAdapter(PlatformConfig(enabled=True))
    binding = posture.make_binding("alice", "", "ctx-1", ["read_file"])
    source = replace(
        adapter.build_source(
            chat_id="ctx-1", user_id="alice", user_name="alice",
        ),
        a2a_mutation_requested=False,
        a2a_mutation_enabled=False,
        a2a_credential_authenticated=True,
        a2a_peer_trusted=True,
        a2a_peer="alice",
        a2a_agent_slug="",
        a2a_context_id="ctx-1",
        a2a_allowed_tool_names=("read_file",),
        a2a_binding=binding,
        a2a_toolset_fingerprint=binding["toolset_fingerprint"],
    )
    unstamped = source
    source = adapter._stamp_ingress_source(source)

    assert not adapter._validate_ingress_source(unstamped)
    assert adapter._validate_ingress_source(source)
    assert source._transport_adapter_ref() is adapter

    copied = SessionSource.from_dict(source.to_dict())
    assert copied._transport_adapter_ref is None
    assert copied._a2a_ingress_capability is None
    assert not adapter._validate_ingress_source(copied)

    changed = replace(source, a2a_peer="mallory")
    assert not adapter._validate_ingress_source(changed)


def test_gateway_accepts_only_capability_from_its_registered_adapter():
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import Platform, PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    registered = A2AAdapter(PlatformConfig(enabled=True))
    foreign = A2AAdapter(PlatformConfig(enabled=True))
    binding = posture.make_binding("alice", "", "ctx-live", ["read_file"])

    def source_for(adapter):
        source = replace(
            adapter.build_source(chat_id="ctx-live", user_id="alice"),
            a2a_mutation_requested=False,
            a2a_mutation_enabled=False,
            a2a_credential_authenticated=True,
            a2a_peer_trusted=True,
            a2a_peer="alice",
            a2a_agent_slug="",
            a2a_context_id="ctx-live",
            a2a_allowed_tool_names=("read_file",),
            a2a_binding=binding,
            a2a_toolset_fingerprint=binding["toolset_fingerprint"],
        )
        return adapter._stamp_ingress_source(source)

    class Runner(GatewayAuthorizationMixin):
        pass

    runner = Runner()
    runner.adapters = {Platform.A2A: registered}
    runner._profile_adapters = {}
    assert runner._validate_a2a_ingress_source(source_for(registered))
    assert not runner._validate_a2a_ingress_source(source_for(foreign))


def test_adapter_readonly_rebind_mints_a_fresh_valid_capability():
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    adapter = A2AAdapter(PlatformConfig(enabled=True))
    binding = posture.make_binding(
        "alice", "", "ctx-1", ["read_file", "terminal"],
        mutation_enabled=True,
    )
    source = replace(
        adapter.build_source(chat_id="ctx-1", user_id="alice"),
        a2a_mutation_requested=True,
        a2a_mutation_enabled=True,
        a2a_credential_authenticated=True,
        a2a_peer_trusted=True,
        a2a_peer="alice",
        a2a_agent_slug="",
        a2a_context_id="ctx-1",
        a2a_mutable_toolsets=("terminal",),
        a2a_allowed_tool_names=("a2a_history", "a2a_list", "read_file", "terminal"),
        a2a_binding=binding,
        a2a_toolset_fingerprint=binding["toolset_fingerprint"],
    )
    source = adapter._stamp_ingress_source(source)
    old_capability = source._a2a_ingress_capability

    narrowed = adapter._rebind_readonly_source(source)

    assert narrowed is not None
    assert narrowed.a2a_mutation_enabled is False
    assert "terminal" not in narrowed.a2a_allowed_tool_names
    assert "a2a_history" not in narrowed.a2a_allowed_tool_names
    assert "a2a_list" not in narrowed.a2a_allowed_tool_names
    assert narrowed._a2a_ingress_capability is not old_capability
    assert adapter._validate_ingress_source(narrowed)


def test_remote_a2a_history_and_list_are_limited_to_bound_context(monkeypatch):
    from plugins.platforms.a2a import tools as a2a_tools

    binding = posture.make_binding("alice", "research", "ctx-own", ["a2a_history"])
    monkeypatch.setattr(a2a_tools, "_load_config", lambda: {})
    monkeypatch.setattr(
        a2a_tools.protocol, "list_conversations", lambda **_kwargs: ["ctx-own", "ctx-other"]
    )
    monkeypatch.setattr(
        a2a_tools.protocol,
        "load_conversation",
        lambda context_id, limit=50, **_kwargs: [{"role": "user", "text": context_id}],
    )

    listing = a2a_tools.a2a_list({}, a2a_binding=binding)
    assert "ctx-own" in listing
    assert "ctx-other" not in listing
    denied = a2a_tools.a2a_history(
        {"context_id": "ctx-other"}, a2a_binding=binding,
    )
    assert "limited to the authenticated bound context" in denied
    allowed = a2a_tools.a2a_history(
        {"context_id": "ctx-own"}, a2a_binding=binding,
    )
    assert "ctx-own" in allowed


def test_remote_history_is_scoped_by_peer_route_and_context(monkeypatch, tmp_path):
    from plugins.platforms.a2a import protocol, tools as a2a_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(a2a_tools, "_load_config", lambda: {})
    protocol.persist_message(
        "ctx-shared", "user", "research secret", "task-r",
        peer="alice", agent_slug="research",
    )
    protocol.persist_message(
        "ctx-shared", "user", "dev secret", "task-d",
        peer="alice", agent_slug="dev",
    )
    research = posture.make_binding(
        "alice", "research", "ctx-shared", ["a2a_history"],
    )
    dev = posture.make_binding(
        "alice", "dev", "ctx-shared", ["a2a_history"],
    )

    research_out = a2a_tools.a2a_history(
        {"context_id": "ctx-shared"}, a2a_binding=research,
    )
    dev_out = a2a_tools.a2a_history(
        {"context_id": "ctx-shared"}, a2a_binding=dev,
    )

    assert "research secret" in research_out
    assert "dev secret" not in research_out
    assert "dev secret" in dev_out
    assert "research secret" not in dev_out


def test_task_store_control_plane_is_peer_scoped():
    from plugins.platforms.a2a import protocol

    store = protocol.TaskStore()
    store.create("task-1", "ctx-1", "alice", "research", "tenant")
    assert store.get("task-1", "research", "tenant", "alice") is not None
    assert store.get("task-1", "research", "tenant", "mallory") is None
    records, _ = store.list(agent_slug="research", tenant="tenant", peer="mallory")
    assert records == []
    assert store.set_push_config(
        "task-1", "https://example.test/push", "research", "tenant", "mallory"
    ) is None


def test_served_agent_peer_binding_is_explicit_not_slug_equality():
    from plugins.platforms.a2a import security

    route = {"slug": "meshfleet", "allowed_peers": ["conductor"]}
    assert security.is_authorized_for_agent("conductor", route)
    assert not security.is_authorized_for_agent("researcher", route)
    assert security.is_authorized_for_agent(
        "researcher", {"slug": "meshfleet", "allowed_peers": []},
    )
    assert not security.is_authorized_for_agent("conductor", None)


def test_prepare_task_establishes_authorized_mutable_binding_and_rejects_reuse(
    monkeypatch, tmp_path,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    monkeypatch.setattr(posture, "load_persisted_bindings", lambda: {})
    monkeypatch.setattr(
        posture, "_binding_store_path", lambda: tmp_path / "bindings.json",
    )
    monkeypatch.setattr(protocol, "load_conversation", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter_module.security, "audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "terminal"}},
            {"type": "function", "function": {"name": "execute_code"}},
            {"type": "function", "function": {"name": "delegate_task"}},
        ],
    )
    adapter = adapter_module.A2AAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "agents": {
                    "research": {
                        "profile": "research",
                        "tenant": "research",
                        "mutable_toolsets": [
                            "terminal", "code_execution", "delegation",
                        ],
                        "mutable_tool_names": [
                            "terminal", "execute_code", "delegate_task",
                        ],
                        "mutation_allowed_peers": ["alice"],
                    }
                }
            },
        )
    )
    captured = {}

    def fake_forward(*_args, posture_policy=None, **_kwargs):
        captured.update(posture_policy or {})
        return "ok", protocol.STATE_COMPLETED

    adapter._forward_to_profile = fake_forward
    params = {
        "message": protocol.text_message(
            protocol.ROLE_USER, "authorized task", context_id="ctx-new",
        )
    }
    params["message"]["metadata"] = {posture.MUTATION_METADATA_KEY: True}
    terminal, pending = adapter._prepare_task(
        params,
        "alice",
        agent=adapter._agents["research"],
        credential_authenticated=True,
    )
    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_COMPLETED
    assert captured["mutation_enabled"] is True
    assert "terminal" in captured["allowed_tool_names"]
    assert "execute_code" not in captured["allowed_tool_names"]
    assert "delegate_task" not in captured["allowed_tool_names"]

    terminal, pending = adapter._prepare_task(
        params,
        "mallory",
        agent=adapter._agents["research"],
        credential_authenticated=True,
    )
    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_REJECTED


def test_invalid_mutation_metadata_has_no_adapter_side_effects(monkeypatch, tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    monkeypatch.setattr(posture, "load_persisted_bindings", lambda: {})
    monkeypatch.setattr(
        posture, "_binding_store_path", lambda: tmp_path / "bindings.json",
    )
    adapter = adapter_module.A2AAdapter(PlatformConfig(enabled=True, extra={}))

    def side_effect(*_args, **_kwargs):
        raise AssertionError("invalid posture reached a side-effect boundary")

    monkeypatch.setattr(posture, "claim_persisted_binding", side_effect)
    monkeypatch.setattr(adapter_module.security, "audit", side_effect)
    monkeypatch.setattr(protocol, "persist_message", side_effect)
    monkeypatch.setattr(adapter.tasks, "create", side_effect)
    params = {
        "message": {
            "role": protocol.ROLE_USER,
            "parts": [{"text": "bad", "mediaType": "text/plain"}],
            "metadata": {posture.MUTATION_METADATA_KEY: "true"},
        }
    }
    terminal, pending = adapter._prepare_task(params, "alice")
    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_REJECTED


def test_binding_claim_is_cross_process_compare_and_set(tmp_path):
    candidate_a = posture.make_binding(
        "alice", "research", "ctx-race", ["read_file"],
    )
    candidate_b = posture.make_binding(
        "mallory", "research", "ctx-race", ["read_file"],
    )
    script = """
import json, sys
from plugins.platforms.a2a import posture
candidate = json.loads(sys.argv[1])
print(json.dumps(posture.claim_persisted_binding('research', 'ctx-race', candidate)))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, json.dumps(candidate)],
            cwd=os.getcwd(), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        for candidate in (candidate_a, candidate_b)
    ]
    results = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0, stderr
        results.append(json.loads(stdout.strip())[
            0
        ])
    assert sorted(results) == ["claimed", "mismatch"]


def test_child_issuer_key_creation_is_cross_process_safe(tmp_path):
    policy = {
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-key-race",
        "mutation_enabled": False,
        "allowed_tool_names": ["read_file"],
        "binding": posture.make_binding(
            "alice", "research", "ctx-key-race", ["read_file"],
        ),
    }
    script = """
import json, sys
from plugins.platforms.a2a import posture
print(json.dumps(posture.sign_child_policy(json.loads(sys.argv[1]))))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, json.dumps(policy)],
            cwd=os.getcwd(), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        for _ in range(8)
    ]
    signatures = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0, stderr
        signatures.append(json.loads(stdout)["signature"])
    assert len(set(signatures)) == 1
    key_path = tmp_path / "hermes-home" / "a2a_child_issuer.key"
    assert len(key_path.read_bytes()) == 32


def test_posture_lock_uses_windows_byte_range_fallback(monkeypatch, tmp_path):
    calls = []

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, mode, size):
            calls.append((fd, mode, size, os.lseek(fd, 0, os.SEEK_CUR)))

    lock_path = tmp_path / "posture.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        monkeypatch.setattr(posture, "fcntl", None)
        monkeypatch.setattr(posture, "msvcrt", FakeMsvcrt)

        posture._lock_fd(fd)
        posture._unlock_fd(fd)

        assert lock_path.read_bytes() == b" "
        assert [(mode, size, offset) for _, mode, size, offset in calls] == [
            (FakeMsvcrt.LK_LOCK, 1, 0),
            (FakeMsvcrt.LK_UNLCK, 1, 0),
        ]
    finally:
        os.close(fd)


def test_forwarded_cli_rejects_missing_policy_before_other_initialization(monkeypatch):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    fake = SimpleNamespace(agent=None)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", " A2A ")
    monkeypatch.delenv(posture.CHILD_POLICY_ENV, raising=False)
    # The intentionally tiny fake proves the method returns before touching
    # credentials, MCP startup, session restoration, or callbacks.
    assert CLIAgentSetupMixin._init_agent(fake) is False
