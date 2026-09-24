"""Multiplex allow-all / allowlist reads must not inherit process env.

A secondary profile with no allowlist of its own used to fall through to
``os.environ`` after a scoped miss. Under multiplex that process env holds
the default profile's ``GATEWAY_ALLOW_ALL_USERS`` / allowlists, so the
secondary gate opened. These tests pin the fail-closed contract: a scoped
miss or empty value denies, and a value present in the scope still wins.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class _Scope:
    def __init__(self, mapping):
        self.mapping = mapping
        self.token = None

    def __enter__(self):
        self.token = ss.set_secret_scope(self.mapping)
        return self

    def __exit__(self, *exc):
        ss.reset_secret_scope(self.token)


def _open_policy_runner():
    """Runner whose secondary adapter forwards everyone at intake.

    Own-policy "open" is not authorization, so the env allow-all / allowlist
    checks are what decide. That is the path that used to borrow process env.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    secondary = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="open",
        _group_policy="open",
    )
    runner.adapters = {}
    runner._profile_adapters = {"coder": {Platform.SLACK: secondary}}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _slack_dm(user_id="U_STRANGER", profile="coder"):
    return SessionSource(
        platform=Platform.SLACK,
        user_id=user_id,
        chat_id="D1",
        user_name=user_id,
        chat_type="dm",
        profile=profile,
    )


class TestGatewayAllowGates:
    def test_scoped_miss_does_not_inherit_allow_all(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")
        runner = _open_policy_runner()
        ss.set_multiplex_active(True)
        with _Scope({}):
            assert runner._is_user_authorized(_slack_dm()) is False

    def test_empty_scoped_allow_all_does_not_fall_through(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        runner = _open_policy_runner()
        ss.set_multiplex_active(True)
        with _Scope({"GATEWAY_ALLOW_ALL_USERS": ""}):
            assert runner._is_user_authorized(_slack_dm()) is False

    def test_scoped_miss_does_not_inherit_allowlist(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_OWNER")
        monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")
        runner = _open_policy_runner()
        ss.set_multiplex_active(True)
        with _Scope({}):
            assert runner._is_user_authorized(_slack_dm("U_OWNER")) is False

    def test_scoped_allowlist_is_authoritative(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_OWNER")
        runner = _open_policy_runner()
        ss.set_multiplex_active(True)
        with _Scope({"GATEWAY_ALLOWED_USERS": "U_OTHER"}):
            assert runner._is_user_authorized(_slack_dm("U_OWNER")) is False
            assert runner._is_user_authorized(_slack_dm("U_OTHER")) is True

    def test_scoped_allow_all_still_opens(self):
        runner = _open_policy_runner()
        ss.set_multiplex_active(True)
        with _Scope({"GATEWAY_ALLOW_ALL_USERS": "true"}):
            assert runner._is_user_authorized(_slack_dm()) is True

    def test_single_profile_environ_unchanged(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        runner = _open_policy_runner()
        assert runner._is_user_authorized(_slack_dm(profile=None)) is True

    def test_unscoped_multiplex_keeps_process_env(self, monkeypatch):
        # Default-profile startup has no scope installed; process env is its own.
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_OWNER")
        runner = _open_policy_runner()
        ss.set_multiplex_active(True)
        assert runner._is_user_authorized(_slack_dm("U_OWNER", profile=None)) is True


class TestSlackInteractiveFallback:
    def _adapter(self):
        from plugins.platforms.slack.adapter import SlackAdapter

        return object.__new__(SlackAdapter)

    def test_scoped_miss_does_not_open_allow_all(self, monkeypatch):
        monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        with _Scope({}):
            assert adapter._is_interactive_user_authorized("U_STRANGER") is False

    def test_scoped_miss_does_not_inherit_allowlist(self, monkeypatch):
        monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_OWNER")
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        with _Scope({}):
            assert adapter._is_interactive_user_authorized("U_OWNER") is False

    def test_scoped_allowlist_authorizes_only_its_members(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_OWNER")
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        with _Scope({"SLACK_ALLOWED_USERS": "U_OK"}):
            assert adapter._is_interactive_user_authorized("U_OK") is True
            assert adapter._is_interactive_user_authorized("U_OWNER") is False

    def test_single_profile_environ_unchanged(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._adapter()
        assert adapter._is_interactive_user_authorized("U_STRANGER") is True

    def test_platform_allow_all_precedes_allowlist(self, monkeypatch):
        monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")
        adapter = self._adapter()
        assert adapter._is_interactive_user_authorized("U_STRANGER") is True


class TestFeishuAllowGates:
    def test_load_settings_scoped_miss_does_not_inherit(self, monkeypatch):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
        monkeypatch.setenv("FEISHU_GROUP_POLICY", "open")
        monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_default")
        ss.set_multiplex_active(True)
        with _Scope({}):
            settings = FeishuAdapter._load_settings(extra={})
        assert settings.allow_bots == "none"
        assert settings.group_policy == "allowlist"
        assert settings.allowed_group_users == frozenset()

    def test_load_settings_scoped_values_win(self, monkeypatch):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_default")
        monkeypatch.setenv("FEISHU_GROUP_POLICY", "open")
        ss.set_multiplex_active(True)
        with _Scope({
            "FEISHU_ALLOW_BOTS": "mentions",
            "FEISHU_GROUP_POLICY": "disabled",
            "FEISHU_ALLOWED_USERS": "ou_profile",
        }):
            settings = FeishuAdapter._load_settings(extra={})
        assert settings.allow_bots == "mentions"
        assert settings.group_policy == "disabled"
        assert settings.allowed_group_users == frozenset({"ou_profile"})

    def test_admit_scoped_miss_does_not_open_dm(self, monkeypatch):
        from tests.gateway.feishu_helpers import make_adapter_skeleton, make_message, make_sender

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("FEISHU_ALLOW_ALL_USERS", "true")
        adapter = make_adapter_skeleton()
        adapter._allowed_group_users = frozenset({"ou_owner"})
        ss.set_multiplex_active(True)
        with _Scope({}):
            reason = adapter._admit(
                make_sender(open_id="ou_stranger"),
                make_message(chat_type="p2p"),
            )
        assert reason == "dm_policy_rejected"

    def test_admit_scoped_allow_all_still_opens(self):
        from tests.gateway.feishu_helpers import make_adapter_skeleton, make_message, make_sender

        adapter = make_adapter_skeleton()
        adapter._allowed_group_users = frozenset({"ou_owner"})
        ss.set_multiplex_active(True)
        with _Scope({"GATEWAY_ALLOW_ALL_USERS": "true"}):
            assert adapter._admit(
                make_sender(open_id="ou_stranger"),
                make_message(chat_type="p2p"),
            ) is None

    def test_single_profile_environ_unchanged(self, monkeypatch):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_env")
        monkeypatch.setenv("FEISHU_GROUP_POLICY", "open")
        settings = FeishuAdapter._load_settings(extra={})
        assert "ou_env" in settings.allowed_group_users
        assert settings.group_policy == "open"


class TestMatrixAllowGates:
    def _config(self):
        return PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )

    def test_init_scoped_miss_does_not_inherit_allowlists(self, monkeypatch):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!default:example.org")
        ss.set_multiplex_active(True)
        with _Scope({}):
            adapter = MatrixAdapter(self._config())
        assert adapter._allowed_user_ids == set()
        assert adapter._allowed_rooms == set()

    def test_init_scoped_allowlist_wins(self, monkeypatch):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        ss.set_multiplex_active(True)
        with _Scope({"MATRIX_ALLOWED_USERS": "@profile:example.org"}):
            adapter = MatrixAdapter(self._config())
        assert adapter._allowed_user_ids == {"@profile:example.org"}

    def test_invite_scoped_miss_does_not_open_allow_all(self, monkeypatch):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = object.__new__(MatrixAdapter)
        adapter._allowed_user_ids = set()
        joined = []
        adapter._schedule_invite_join = lambda *args, **kwargs: joined.append(args)
        event = SimpleNamespace(
            room_id="!room:example.org",
            sender="@stranger:example.org",
            content=SimpleNamespace(is_direct=False),
        )
        ss.set_multiplex_active(True)
        with _Scope({}):
            asyncio.run(adapter._on_invite(event))
        assert joined == []

    def test_invite_scoped_allow_all_still_joins(self):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = object.__new__(MatrixAdapter)
        adapter._allowed_user_ids = set()
        joined = []
        adapter._schedule_invite_join = lambda *args, **kwargs: joined.append(args)
        event = SimpleNamespace(
            room_id="!room:example.org",
            sender="@owner:example.org",
            content=SimpleNamespace(is_direct=False),
        )
        ss.set_multiplex_active(True)
        with _Scope({"GATEWAY_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._on_invite(event))
        assert joined

    def test_reactor_scoped_miss_does_not_open_allow_all(self, monkeypatch):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = object.__new__(MatrixAdapter)
        adapter._allowed_user_ids = set()
        adapter._approval_require_sender = False
        adapter._send_invalid_reaction_feedback = AsyncMock()
        prompt = SimpleNamespace(requester_user_id="@owner:example.org")
        ss.set_multiplex_active(True)
        with _Scope({}):
            allowed = asyncio.run(
                adapter._validate_matrix_prompt_reactor(
                    "!room:example.org",
                    "$event",
                    "@stranger:example.org",
                    prompt,
                    "approval",
                )
            )
        assert allowed is False
        adapter._send_invalid_reaction_feedback.assert_awaited()

    def test_single_profile_environ_unchanged(self, monkeypatch):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@env:example.org")
        adapter = MatrixAdapter(self._config())
        assert adapter._allowed_user_ids == {"@env:example.org"}
