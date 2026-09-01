from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import posture


def _binding(
    peer,
    agent_slug,
    context_id,
    tool_names,
    *,
    served_profile=None,
    served_tenant=None,
    profile_home_identity=None,
    mutation_enabled=False,
):
    """Build an explicitly route-bound posture value for unit fixtures."""
    profile = served_profile or agent_slug or "default"
    tenant = agent_slug if served_tenant is None else served_tenant
    home_identity = profile_home_identity or f"test-home:{profile}"
    return posture.make_binding(
        peer,
        agent_slug,
        context_id,
        tool_names,
        served_profile=profile,
        served_tenant=tenant,
        profile_home_identity=home_identity,
        mutation_enabled=mutation_enabled,
    )


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
    binding = _binding(
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

    binding = _binding(
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
        True, ["terminal", "execute_code", "delegate_task", "tool_call"],
    )

    assert "terminal" in allowed
    assert "execute_code" not in allowed
    assert "delegate_task" not in allowed
    assert "tool_call" not in allowed


def test_resume_binding_mismatch_rejects_but_corrupt_binding_is_read_only():
    expected = _binding("alice", "research", "ctx-1", ["read_file"])
    assert posture.resume_binding_status(expected, expected) == "ok"
    other = _binding("alice", "research", "ctx-1", ["terminal"])
    assert posture.resume_binding_status(expected, other) == "mismatch"
    assert posture.resume_binding_status({"peer": "alice"}, expected) == "corrupt"


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("served_profile", "admin"),
        ("served_tenant", "tenant-b"),
        ("profile_home_identity", "home-b"),
    ],
)
def test_resume_binding_rejects_same_slug_route_identity_remap(
    changed_field, changed_value,
):
    base = _binding(
        "alice", "research", "ctx-remap", ["terminal"],
        mutation_enabled=True,
    )
    original = {
        **base,
        "served_profile": "research",
        "served_tenant": "tenant-a",
        "profile_home_identity": "home-a",
    }
    remapped = {**original, changed_field: changed_value}

    assert posture.resume_binding_status(original, remapped) == "mismatch"


def test_profile_home_identity_changes_when_profile_is_recreated(tmp_path):
    profile_home = tmp_path / "profiles" / "research"
    profile_home.mkdir(parents=True)

    original = posture.profile_home_identity(str(profile_home))
    assert posture.profile_home_identity(str(profile_home)) == original

    shutil.rmtree(profile_home)
    profile_home.mkdir()

    assert posture.profile_home_identity(str(profile_home)) != original


def test_profile_home_identity_serializes_concurrent_first_creation(
    monkeypatch, tmp_path
):
    profile_home = tmp_path / "profiles" / "research"
    profile_home.mkdir(parents=True)
    identity_path = profile_home / posture.PROFILE_INSTANCE_ID_FILE
    real_open = posture.os.open
    real_write = posture.os.write
    creator_ready = threading.Event()
    release_creator = threading.Event()
    identity_fds = set()
    results = []
    errors = []

    def tracked_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if Path(path) == identity_path and flags & os.O_EXCL:
            identity_fds.add(fd)
        return fd

    def delayed_identity_write(fd, payload):
        if fd in identity_fds and not creator_ready.is_set():
            creator_ready.set()
            assert release_creator.wait(timeout=2)
        return real_write(fd, payload)

    def resolve_identity():
        try:
            results.append(posture.profile_home_identity(str(profile_home)))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(posture.os, "open", tracked_open)
    monkeypatch.setattr(posture.os, "write", delayed_identity_write)

    creator = threading.Thread(target=resolve_identity)
    creator.start()
    assert creator_ready.wait(timeout=2)

    loser = threading.Thread(target=resolve_identity)
    loser.start()
    loser.join(timeout=0.1)
    release_creator.set()
    creator.join(timeout=2)
    loser.join(timeout=2)

    assert not creator.is_alive()
    assert not loser.is_alive()
    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    if os.name != "nt":
        assert identity_path.stat().st_mode & 0o777 == 0o600
        lock_path = profile_home / posture.PROFILE_INSTANCE_LOCK_FILE
        assert lock_path.stat().st_mode & 0o777 == 0o600


def test_profile_home_identity_rejects_legacy_path_hash_binding(tmp_path):
    profile_home = tmp_path / "profiles" / "research"
    profile_home.mkdir(parents=True)
    legacy_path_hash = hashlib.sha256(
        str(profile_home.resolve()).encode("utf-8"),
    ).hexdigest()
    legacy = _binding(
        "alice", "research", "ctx-legacy-home", ["terminal"],
        profile_home_identity=legacy_path_hash,
        mutation_enabled=True,
    )
    current = {
        **legacy,
        "profile_home_identity": posture.profile_home_identity(str(profile_home)),
    }

    assert posture.resume_binding_status(legacy, current) == "mismatch"


def test_forwarded_child_rejects_recreated_profile_instance(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "research"
    profile_home.mkdir(parents=True)
    old_identity = posture.profile_home_identity(str(profile_home))
    binding = _binding(
        "alice",
        "research",
        "ctx-recreated-child",
        ["terminal"],
        profile_home_identity=old_identity,
        mutation_enabled=True,
    )
    secret = b"r" * 32
    signed = posture.sign_child_policy(
        {
            "authenticated": True,
            "served_agent_slug": "research",
            "context_id": "ctx-recreated-child",
            "mutation_enabled": True,
            "allowed_tool_names": ["terminal"],
            "binding": binding,
        },
        secret=secret,
    )

    shutil.rmtree(profile_home)
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv(posture.CHILD_POLICY_ENV, json.dumps(signed))

    loaded = posture.load_child_policy(secret=secret)

    assert loaded and "profile instance identity mismatch" in loaded["error"]


def test_forwarded_child_rechecks_profile_instance_at_launch(monkeypatch, tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    adapter = adapter_module.A2AAdapter(PlatformConfig(enabled=True))
    profile_home = tmp_path / "profiles" / "research"
    profile_home.mkdir(parents=True)
    identity = posture.profile_home_identity(str(profile_home))
    binding = _binding(
        "alice",
        "research",
        "ctx-launch-replacement",
        ["terminal"],
        served_tenant="",
        profile_home_identity=identity,
        mutation_enabled=True,
    )
    policy = {
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-launch-replacement",
        "mutation_enabled": True,
        "allowed_tool_names": ["terminal"],
        "served_profile": "research",
        "served_tenant": "",
        "profile_home_identity": identity,
        "binding": binding,
    }
    secret = b"l" * 32
    real_sign = posture.sign_child_policy
    child_result = {}

    def sign_with_test_key(value):
        return real_sign(value, secret=secret)

    def launch_after_replacement(_cmd, _timeout, env):
        shutil.rmtree(profile_home)
        profile_home.mkdir()
        with monkeypatch.context() as child_env:
            child_env.setenv("HERMES_HOME", env["HERMES_HOME"])
            child_env.setenv(
                posture.CHILD_POLICY_ENV, env[posture.CHILD_POLICY_ENV]
            )
            child_result["policy"] = posture.load_child_policy(secret=secret)
        return 1, "", "child rejected stale profile authority"

    monkeypatch.setattr(
        adapter_module, "_profile_home", lambda _profile: str(profile_home)
    )
    monkeypatch.setattr(posture, "sign_child_policy", sign_with_test_key)
    monkeypatch.setattr(adapter, "_lookup_forward_session", lambda *_args: None)
    monkeypatch.setattr(adapter, "_latest_a2a_session", lambda *_args: None)
    monkeypatch.setattr(adapter, "_run_profile_command", launch_after_replacement)

    reply, state = adapter._forward_to_profile(
        {"slug": "research", "profile": "research"},
        "alice",
        "ctx-launch-replacement",
        "hello",
        posture_policy=policy,
    )

    assert state == protocol.STATE_FAILED
    assert "profile instance identity mismatch" in child_result["policy"]["error"]
    assert "failed rc=1" in reply


def test_agent_cache_key_rejects_same_slug_profile_remap():
    original = _binding(
        "alice", "research", "ctx-cache", ["read_file"],
        served_profile="research",
        served_tenant="tenant-a",
        profile_home_identity="home-a",
    )
    source = SimpleNamespace(
        a2a_binding=original,
        a2a_peer="alice",
        a2a_agent_slug="research",
        a2a_context_id="ctx-cache",
    )
    remapped = SimpleNamespace(
        **vars(source),
    )
    remapped.a2a_binding = {**original, "served_profile": "admin"}

    assert posture.request_key(source) != posture.request_key(remapped)


def test_pre_identity_binding_is_legacy_not_current(tmp_path, monkeypatch):
    path = tmp_path / "bindings.json"
    monkeypatch.setattr(posture, "_binding_store_path", lambda: path)
    current = _binding(
        "alice", "research", "ctx-legacy", ["terminal"],
        mutation_enabled=True,
    )
    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"served_profile", "served_tenant", "profile_home_identity"}
    }
    path.write_text(
        json.dumps({"research\u001fctx-legacy": legacy}),
        encoding="utf-8",
    )
    candidate = current

    status, persisted = posture.claim_persisted_binding(
        "research", "ctx-legacy", candidate,
    )

    assert status == "legacy"
    assert persisted == legacy
    assert json.loads(path.read_text(encoding="utf-8"))[
        "research\u001fctx-legacy"
    ] == legacy


def test_binding_store_is_atomic_and_round_trips(monkeypatch, tmp_path):
    path = tmp_path / "bindings.json"
    monkeypatch.setattr(posture, "_binding_store_path", lambda: path)
    binding = _binding("alice", "research", "ctx-1", ["read_file"])
    posture.persist_bindings({("research", "ctx-1"): binding})
    loaded = posture.load_persisted_bindings()
    assert loaded[("research", "ctx-1")] == binding


def test_forwarded_child_policy_is_consumed_before_tool_dispatch(monkeypatch, tmp_path):
    secret = b"a" * 32
    key_dir = tmp_path / "hermes-home"
    key_dir.mkdir(mode=0o700)
    binding = _binding(
        "alice",
        "research",
        "ctx-1",
        ["read_file"],
        profile_home_identity=posture.profile_home_identity(str(key_dir)),
    )
    unsigned_policy = {
        "authenticated": True,
        "served_agent_slug": "research",
        "context_id": "ctx-1",
        "mutation_enabled": False,
        "allowed_tool_names": ["read_file"],
        "binding": binding,
    }
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
    secret = b"z" * 32
    key_dir = tmp_path / "hermes-home"
    key_dir.mkdir(mode=0o700)
    binding = _binding(
        "alice",
        "research",
        "ctx-zero",
        ["read_file"],
        profile_home_identity=posture.profile_home_identity(str(key_dir)),
    )
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
    mutable_binding = _binding(
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

    valid_binding = _binding(
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


def test_forwarded_policy_rejects_route_identity_mismatch():
    secret = b"r" * 32
    binding = {
        **_binding(
            "alice", "research", "ctx-route", ["read_file"],
        ),
        "served_profile": "research",
        "served_tenant": "tenant-a",
        "profile_home_identity": "home-a",
    }
    policy = posture.sign_child_policy(
        {
            "authenticated": True,
            "served_agent_slug": "research",
            "served_profile": "admin",
            "served_tenant": "tenant-a",
            "profile_home_identity": "home-a",
            "context_id": "ctx-route",
            "mutation_enabled": False,
            "allowed_tool_names": ["read_file"],
            "binding": binding,
        },
        secret=secret,
    )

    assert posture.load_child_policy(policy, secret=secret).get("error")


def test_forwarded_child_rejects_profile_home_remap_before_launch(monkeypatch, tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    adapter = adapter_module.A2AAdapter(PlatformConfig(enabled=True))
    agent = {"slug": "research", "profile": "research", "tenant": "tenant-a"}
    old_home = tmp_path / "profiles" / "old-research"
    old_home.mkdir(parents=True)
    old_home_identity = posture.profile_home_identity(str(old_home))
    binding = _binding(
        "alice", "research", "ctx-route", ["read_file"],
        served_profile="research",
        served_tenant="tenant-a",
        profile_home_identity=old_home_identity,
    )
    policy = {
        "served_profile": "research",
        "served_tenant": "tenant-a",
        "profile_home_identity": old_home_identity,
        "binding": binding,
    }
    monkeypatch.setattr(
        adapter_module,
        "_profile_home",
        lambda _profile: str(tmp_path / "profiles" / "new-research"),
    )
    monkeypatch.setattr(
        adapter, "_run_profile_command",
        lambda *_args, **_kwargs: pytest.fail("remapped child must not launch"),
    )

    reply, state = adapter._forward_to_profile(
        agent, "alice", "ctx-route", "hello", posture_policy=policy,
    )

    assert state == protocol.STATE_FAILED
    assert "identity changed" in reply


def test_forwarded_child_rechecks_profile_identity_after_waiting_for_lock(
    monkeypatch, tmp_path,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    adapter = adapter_module.A2AAdapter(PlatformConfig(enabled=True))
    agent = {"slug": "research", "profile": "research", "tenant": "tenant-a"}
    old_home = tmp_path / "profiles" / "old-research"
    new_home = tmp_path / "profiles" / "new-research"
    old_home.mkdir(parents=True)
    new_home.mkdir()
    current_home = {"path": str(old_home)}
    old_home_identity = posture.profile_home_identity(str(old_home))
    binding = _binding(
        "alice", "research", "ctx-lock-remap", ["read_file"],
        served_profile="research",
        served_tenant="tenant-a",
        profile_home_identity=old_home_identity,
    )
    policy = {
        "served_profile": "research",
        "served_tenant": "tenant-a",
        "profile_home_identity": old_home_identity,
        "binding": binding,
    }

    class RemappingLock:
        def __enter__(self):
            current_home["path"] = str(new_home)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        adapter_module, "_profile_home", lambda _profile: current_home["path"],
    )
    monkeypatch.setattr(adapter, "_forward_lock", lambda _profile: RemappingLock())
    monkeypatch.setattr(
        adapter, "_run_profile_command",
        lambda *_args, **_kwargs: pytest.fail("remapped child must not launch"),
    )

    reply, state = adapter._forward_to_profile(
        agent, "alice", "ctx-lock-remap", "hello", posture_policy=policy,
    )

    assert state == protocol.STATE_FAILED
    assert "identity changed" in reply


@pytest.mark.parametrize(
    "forbidden_name", ["execute_code", "delegate_task", "tool_call"],
)
def test_forwarded_policy_rejects_unbounded_composite_tools(forbidden_name):
    secret = b"c" * 32
    binding = _binding(
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


@pytest.mark.parametrize(
    "forbidden_name", ["execute_code", "delegate_task", "tool_call"],
)
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


@pytest.mark.parametrize(
    "forbidden_name", ["execute_code", "delegate_task", "tool_call"],
)
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
    binding = _binding("alice", "research", "ctx-1", ["read_file"])
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


@pytest.mark.parametrize(
    "forbidden_name", ["execute_code", "delegate_task", "tool_call"],
)
def test_source_binding_rejects_unbounded_composite_tools(forbidden_name):
    binding = _binding(
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
    binding = _binding("alice", "", "ctx-1", ["read_file"])
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
    binding = _binding("alice", "", "ctx-live", ["read_file"])

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
    binding = _binding(
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

    binding = _binding("alice", "research", "ctx-own", ["a2a_history"])
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
    research = _binding(
        "alice", "research", "ctx-shared", ["a2a_history"],
    )
    dev = _binding(
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


def test_served_profile_toolset_scope_fails_closed_without_valid_config(
    monkeypatch, tmp_path,
):
    from plugins.platforms.a2a import adapter as adapter_module

    profile_home = tmp_path / "research"
    profile_home.mkdir()
    monkeypatch.setattr(adapter_module, "_profile_home", lambda _profile: str(profile_home))
    route = {"profile": "research", "local": False}

    assert adapter_module._served_profile_toolset_scope(route) is None

    (profile_home / "config.yaml").write_text("- malformed\n", encoding="utf-8")
    assert adapter_module._served_profile_toolset_scope(route) is None

    (profile_home / "config.yaml").write_text(
        "platform_toolsets:\n  cli: hermes-cli\n", encoding="utf-8",
    )
    assert adapter_module._served_profile_toolset_scope(route) is None

    (profile_home / "config.yaml").write_text(
        "agent:\n  disabled_toolsets:\n    malformed: mapping\n", encoding="utf-8",
    )
    assert adapter_module._served_profile_toolset_scope(route) is None


def test_local_default_profile_without_config_uses_canonical_scope(monkeypatch, tmp_path):
    from plugins.platforms.a2a import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_profile_home", lambda _profile: str(tmp_path))

    resolved = adapter_module._served_profile_toolset_scope(
        {"profile": "default", "local": True}
    )

    assert resolved is not None
    enabled, disabled = resolved
    assert isinstance(enabled, list)
    assert enabled
    assert disabled == []


def test_forwarded_profile_scope_honors_cli_coding_focus(monkeypatch, tmp_path):
    from agent import coding_context
    from hermes_cli import tools_config
    from plugins.platforms.a2a import adapter as adapter_module

    (tmp_path / "config.yaml").write_text(
        "agent:\n  coding_context: focus\n", encoding="utf-8",
    )
    monkeypatch.setattr(adapter_module, "_profile_home", lambda _profile: str(tmp_path))
    monkeypatch.setattr(
        coding_context,
        "coding_selection",
        lambda **_kwargs: ["coding"],
    )
    monkeypatch.setattr(
        tools_config,
        "_get_platform_tools",
        lambda *_args, **_kwargs: pytest.fail(
            "focus selection must replace the ordinary CLI toolset resolver"
        ),
    )

    assert adapter_module._served_profile_toolset_scope(
        {"profile": "research", "local": False}
    ) == (["coding"], [])


def test_forwarded_profile_scope_uses_canonical_env_expansion_and_defaults(
    monkeypatch, tmp_path,
):
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.tools_config import _get_platform_tools
    from plugins.platforms.a2a import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_profile_home", lambda _profile: str(tmp_path))
    monkeypatch.setenv("A2A_TEST_TOOLSET", "hermes-cli")
    (tmp_path / "config.yaml").write_text(
        'platform_toolsets:\n  cli: ["${A2A_TEST_TOOLSET}"]\n',
        encoding="utf-8",
    )

    expanded = adapter_module._served_profile_toolset_scope(
        {"profile": "research", "local": False}
    )
    expected_config = dict(DEFAULT_CONFIG)
    expected_config["platform_toolsets"] = {"cli": ["hermes-cli"]}
    assert expanded == (sorted(_get_platform_tools(expected_config, "cli")), [])

    (tmp_path / "config.yaml").write_text(
        "agent:\n  max_turns: 7\n",
        encoding="utf-8",
    )
    resolved = adapter_module._served_profile_toolset_scope(
        {"profile": "research", "local": False}
    )

    assert resolved is not None
    enabled, disabled = resolved
    assert enabled == sorted(_get_platform_tools(DEFAULT_CONFIG, "cli"))
    assert disabled == []


def test_malformed_served_scope_does_not_hydrate_profile_secrets(
    monkeypatch, tmp_path,
):
    from hermes_cli import env_loader
    from plugins.platforms.a2a import adapter as adapter_module

    (tmp_path / "config.yaml").write_text(
        "platform_toolsets:\n  cli: hermes-cli\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter_module, "_profile_home", lambda _profile: str(tmp_path))
    hydrate_calls = []
    monkeypatch.setattr(
        env_loader,
        "hydrate_profile_secret_sources",
        lambda home: hydrate_calls.append(home),
    )

    assert adapter_module._served_profile_toolset_scope(
        {"profile": "research", "local": False}
    ) is None
    assert hydrate_calls == []


def test_prepare_task_scopes_binding_catalog_to_served_profile(monkeypatch, tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    profile_home = tmp_path / "research"
    profile_home.mkdir()
    monkeypatch.setattr(
        adapter_module, "_profile_home", lambda _profile: str(profile_home),
    )
    monkeypatch.setattr(posture, "load_persisted_bindings", lambda: {})
    monkeypatch.setattr(
        posture, "_binding_store_path", lambda: tmp_path / "bindings.json",
    )
    monkeypatch.setattr(protocol, "load_conversation", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter_module.security, "audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter_module,
        "_served_profile_toolset_scope",
        lambda _route: (["hermes-cli"], ["a2a"]),
    )
    catalog_calls = []

    def fake_catalog(**kwargs):
        catalog_calls.append(kwargs)
        names = ["read_file"]
        # The raw catalog contains outbound A2A plugin tools, but Tool Search
        # defers them from the child's directly exposed schema.  The signed
        # policy must bind the latter surface.
        if kwargs.get("skip_tool_search_assembly"):
            names += ["a2a_history", "a2a_list"]
        return [
            {"type": "function", "function": {"name": name}}
            for name in names
        ]

    monkeypatch.setattr("model_tools.get_tool_definitions", fake_catalog)
    adapter = adapter_module.A2AAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "agents": {
                    "research": {
                        "profile": "research",
                        "tenant": "research",
                    }
                }
            },
        )
    )
    captured = {}
    protected = tmp_path / "read-only-proof.txt"
    protected.write_text("scoped sentinel", encoding="utf-8")
    tool_observations = {}

    def fake_forward(*_args, posture_policy=None, **_kwargs):
        captured.update(posture_policy or {})
        from model_tools import handle_function_call

        tool_observations["read"] = handle_function_call(
            "read_file", {"path": str(protected)}, a2a_posture=posture_policy,
        )
        tool_observations["terminal"] = handle_function_call(
            "terminal",
            {"command": f"printf changed > {protected}"},
            a2a_posture=posture_policy,
        )
        return "ok", protocol.STATE_COMPLETED

    adapter._forward_to_profile = fake_forward
    params = {
        "message": protocol.text_message(
            protocol.ROLE_USER, "read-only task", context_id="ctx-scoped",
        )
    }

    terminal, pending = adapter._prepare_task(
        params,
        "alice",
        agent=adapter._agents["research"],
        credential_authenticated=True,
    )

    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_COMPLETED
    assert catalog_calls == [{
        "enabled_toolsets": ["hermes-cli"],
        "disabled_toolsets": ["a2a"],
        "quiet_mode": True,
        "skip_tool_search_assembly": False,
    }]
    assert captured["allowed_tool_names"] == ["read_file"]
    assert "a2a_history" not in captured["allowed_tool_names"]
    assert "a2a_list" not in captured["allowed_tool_names"]
    assert posture.child_policy_matches_tools(captured, ["read_file"])
    assert "scoped sentinel" in str(tool_observations["read"])
    assert "A2A mutation posture" in str(tool_observations["terminal"])
    assert protected.read_text(encoding="utf-8") == "scoped sentinel"


def test_prepare_task_rejects_when_served_profile_scope_is_unavailable(
    monkeypatch, tmp_path,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    profile_home = tmp_path / "research"
    profile_home.mkdir()
    monkeypatch.setattr(
        adapter_module, "_profile_home", lambda _profile: str(profile_home),
    )
    monkeypatch.setattr(posture, "load_persisted_bindings", lambda: {})
    monkeypatch.setattr(
        posture, "_binding_store_path", lambda: tmp_path / "bindings.json",
    )
    monkeypatch.setattr(adapter_module, "_served_profile_toolset_scope", lambda _route: None)
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: pytest.fail("unscoped catalog must not be assembled"),
    )
    adapter = adapter_module.A2AAdapter(
        PlatformConfig(
            enabled=True,
            extra={"agents": {"research": {"profile": "research"}}},
        )
    )
    def side_effect(*_args, **_kwargs):
        raise AssertionError("missing scope reached a side-effect boundary")

    monkeypatch.setattr(adapter_module.security, "audit", side_effect)
    monkeypatch.setattr(protocol, "persist_message", side_effect)
    monkeypatch.setattr(posture, "claim_persisted_binding", side_effect)
    monkeypatch.setattr(adapter.tasks, "create", side_effect)
    adapter._forward_to_profile = side_effect
    params = {
        "message": protocol.text_message(
            protocol.ROLE_USER, "must fail closed", context_id="ctx-missing-scope",
        )
    }

    terminal, pending = adapter._prepare_task(
        params,
        "alice",
        agent=adapter._agents["research"],
        credential_authenticated=True,
    )

    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_REJECTED
    assert "served profile tool scope" in terminal["status"]["message"]["parts"][0]["text"]


def test_prepare_task_rejects_when_scoped_catalog_assembly_fails_before_side_effects(
    monkeypatch, tmp_path,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    profile_home = tmp_path / "research"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(
        "platform_toolsets:\n  cli: [hermes-cli]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter_module, "_profile_home", lambda _profile: str(profile_home))
    monkeypatch.setattr(posture, "load_persisted_bindings", lambda: {})
    monkeypatch.setattr(posture, "_binding_store_path", lambda: tmp_path / "bindings.json")
    monkeypatch.setattr(
        adapter_module,
        "_served_profile_toolset_scope",
        lambda _route: (["hermes-cli"], ["a2a"]),
    )
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("catalog unavailable")),
    )
    adapter = adapter_module.A2AAdapter(
        PlatformConfig(
            enabled=True,
            extra={"agents": {"research": {"profile": "research"}}},
        )
    )

    def side_effect(*_args, **_kwargs):
        raise AssertionError("catalog failure reached a side-effect boundary")

    monkeypatch.setattr(adapter_module.security, "audit", side_effect)
    monkeypatch.setattr(protocol, "persist_message", side_effect)
    monkeypatch.setattr(posture, "claim_persisted_binding", side_effect)
    monkeypatch.setattr(adapter.tasks, "create", side_effect)
    adapter._forward_to_profile = side_effect
    params = {
        "message": protocol.text_message(
            protocol.ROLE_USER,
            "must fail closed",
            context_id="ctx-catalog-failure",
        )
    }

    terminal, pending = adapter._prepare_task(
        params,
        "alice",
        agent=adapter._agents["research"],
        credential_authenticated=True,
    )

    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_REJECTED
    assert "catalog" in terminal["status"]["message"]["parts"][0]["text"]


def test_prepare_task_establishes_authorized_mutable_binding_and_rejects_reuse(
    monkeypatch, tmp_path,
):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a import adapter as adapter_module
    from plugins.platforms.a2a import protocol

    profile_home = tmp_path / "research"
    profile_home.mkdir()
    monkeypatch.setattr(
        adapter_module, "_profile_home", lambda _profile: str(profile_home),
    )
    monkeypatch.setattr(posture, "load_persisted_bindings", lambda: {})
    monkeypatch.setattr(
        posture, "_binding_store_path", lambda: tmp_path / "bindings.json",
    )
    monkeypatch.setattr(protocol, "load_conversation", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(protocol, "persist_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter_module.security, "audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter_module,
        "_served_profile_toolset_scope",
        lambda _route: (["hermes-cli"], []),
    )
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "terminal"}},
            {"type": "function", "function": {"name": "execute_code"}},
            {"type": "function", "function": {"name": "delegate_task"}},
            {"type": "function", "function": {"name": "tool_call"}},
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
                            "terminal",
                            "execute_code",
                            "delegate_task",
                            "tool_call",
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
    assert "tool_call" not in captured["allowed_tool_names"]

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
    candidate_a = _binding(
        "alice", "research", "ctx-race", ["read_file"],
    )
    candidate_b = _binding(
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
        "binding": _binding(
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
