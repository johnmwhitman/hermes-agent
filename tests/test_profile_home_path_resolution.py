"""Profile-home resolution for sites that used to ignore get_hermes_home().

Telegram gmail-triage scripts, the Google Chat bot-id cache, and OpenViking's
server log / provider home must follow context override → HERMES_HOME →
platform default — not Path.home() / ".hermes" or a raw os.getenv("HERMES_HOME").
"""

from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def decoy_os_home(tmp_path, monkeypatch):
    decoy = tmp_path / "decoy-os-home"
    decoy.mkdir()
    monkeypatch.setattr(Path, "home", lambda: decoy)
    return decoy


def _under_override(home: Path, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


class TestTelegramGmailTriageScriptPath:
    def test_follows_hermes_home_env(self, profile_home, decoy_os_home):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        path = object.__new__(TelegramAdapter)._gmail_triage_script_path("send-draft.sh")
        assert path == profile_home / "scripts" / "gmail-triage" / "send-draft.sh"
        assert decoy_os_home not in path.parents
        assert path != decoy_os_home / ".hermes" / "scripts" / "gmail-triage" / "send-draft.sh"

    def test_follows_context_override(self, profile_home, decoy_os_home, tmp_path):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        override = tmp_path / "override-home"
        override.mkdir()
        path = _under_override(
            override,
            lambda: object.__new__(TelegramAdapter)._gmail_triage_script_path("archive.sh"),
        )
        assert path == override / "scripts" / "gmail-triage" / "archive.sh"
        assert path != profile_home / "scripts" / "gmail-triage" / "archive.sh"


class TestGoogleChatBotIdCachePath:
    def test_follows_hermes_home_env(self, profile_home, decoy_os_home):
        from plugins.platforms.google_chat.adapter import GoogleChatAdapter

        path = object.__new__(GoogleChatAdapter)._bot_id_cache_path()
        assert path == profile_home / "google_chat_bot_id.json"
        assert decoy_os_home not in path.parents

    def test_follows_context_override(self, profile_home, decoy_os_home, tmp_path):
        from plugins.platforms.google_chat.adapter import GoogleChatAdapter

        override = tmp_path / "override-home"
        override.mkdir()
        path = _under_override(
            override,
            lambda: object.__new__(GoogleChatAdapter)._bot_id_cache_path(),
        )
        assert path == override / "google_chat_bot_id.json"
        assert path != profile_home / "google_chat_bot_id.json"


class TestOpenVikingHermesHome:
    def test_server_log_follows_hermes_home_env(self, profile_home, decoy_os_home):
        from plugins.memory.openviking import _openviking_server_log_path

        path = _openviking_server_log_path()
        assert path == profile_home / "logs" / "openviking-server.log"
        assert decoy_os_home not in path.parents

    def test_server_log_follows_context_override(self, profile_home, decoy_os_home, tmp_path):
        from plugins.memory.openviking import _openviking_server_log_path

        override = tmp_path / "override-home"
        override.mkdir()
        path = _under_override(override, _openviking_server_log_path)
        assert path == override / "logs" / "openviking-server.log"
        assert path != profile_home / "logs" / "openviking-server.log"

    def test_initialize_home_follows_override(self, profile_home, decoy_os_home, tmp_path, monkeypatch):
        import plugins.memory.openviking as openviking_module
        from plugins.memory.openviking import OpenVikingMemoryProvider

        override = tmp_path / "override-home"
        override.mkdir()

        def _blocked_settings(*_a, **_k):
            raise openviking_module._OpenVikingEndpointError("blocked for path test")

        monkeypatch.setattr(
            openviking_module, "_resolve_connection_settings", _blocked_settings
        )
        monkeypatch.setattr(
            OpenVikingMemoryProvider, "_acquire_run_lock", lambda self: None
        )

        provider = OpenVikingMemoryProvider()
        _under_override(override, lambda: provider.initialize("session-path-test"))
        assert provider._hermes_home == str(override)
        assert provider._hermes_home != str(decoy_os_home / ".hermes")
        assert provider._hermes_home != str(profile_home)
