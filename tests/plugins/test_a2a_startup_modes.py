"""A2A listener-mode startup contracts.

Outbound A2A client tools are independent of the inbound HTTP adapter.  A
profile configured only to call remote peers must therefore be able to keep
the plugin enabled without constructing a local listener.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms import a2a as a2a_plugin
from plugins.platforms.a2a import adapter as adapter_module


def _isolated_adapter(monkeypatch, tmp_path, extra: dict) -> adapter_module.A2AAdapter:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    for name in (
        "A2A_PORT",
        "A2A_HOST",
        "A2A_BEARER_TOKEN",
        "A2A_PEER_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    return adapter_module.A2AAdapter(PlatformConfig(enabled=True, extra=extra))


class _ListenerMustNotBeConstructed:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("outbound-only A2A attempted to construct a listener")


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"role": "remote", "port": 0}, id="remote-role"),
        pytest.param({"inbound_disabled": True}, id="inbound-disabled"),
    ],
)
def test_outbound_only_modes_connect_without_constructing_listener(
    monkeypatch,
    tmp_path,
    extra,
):
    adapter = _isolated_adapter(monkeypatch, tmp_path, extra)
    monkeypatch.setattr(adapter_module, "_A2AServer", _ListenerMustNotBeConstructed)

    assert asyncio.run(adapter.connect()) is True
    assert adapter.is_connected is True
    assert adapter._httpd is None
    assert adapter._server_thread is None
    assert adapter._watchdog_thread is None

    asyncio.run(adapter.disconnect())


def test_top_level_remote_config_is_bridged_into_platform_extra():
    registered_tools = set()
    platform_registration = {}

    class Context:
        def register_tool(self, *, name, **_kwargs):
            registered_tools.add(name)

        def register_platform(self, **kwargs):
            platform_registration.update(kwargs)

    a2a_plugin.register(Context())
    raw = {
        "enabled": True,
        "role": "remote",
        "port": 0,
        "extra": {"inbound_disabled": True},
    }

    seeded = platform_registration["apply_yaml_config_fn"]({}, raw)

    assert seeded == {"role": "remote", "port": 0}
    assert registered_tools == {
        "a2a_call",
        "a2a_discover",
        "a2a_history",
        "a2a_list",
        "a2a_orchestrate",
    }
    config = PlatformConfig.from_dict({
        "enabled": True,
        "extra": {**raw["extra"], **seeded},
    })
    assert config.extra == {
        "inbound_disabled": True,
        "role": "remote",
        "port": 0,
    }


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"role": "sideways"}, id="unknown-role"),
        pytest.param({"inbound_disabled": "true"}, id="string-disabled-flag"),
    ],
)
def test_malformed_listener_modes_fail_closed_before_server_construction(
    monkeypatch,
    tmp_path,
    extra,
):
    adapter = _isolated_adapter(monkeypatch, tmp_path, extra)
    monkeypatch.setattr(adapter_module, "_A2AServer", _ListenerMustNotBeConstructed)

    assert asyncio.run(adapter.connect()) is False
    assert adapter.is_connected is False
    assert adapter.fatal_error_code == "invalid_listener_mode"
    assert adapter._httpd is None


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="legacy-default"),
        pytest.param({"role": "inbound"}, id="explicit-inbound"),
        pytest.param({"role": "local"}, id="explicit-local"),
        pytest.param({"inbound_disabled": False}, id="explicit-enabled"),
    ],
)
def test_inbound_modes_still_start_the_http_server(monkeypatch, tmp_path, extra):
    calls = []

    class FakeServer:
        def __init__(self, addr, handler_cls, adapter):
            calls.append(("server", addr, handler_cls, adapter))

        def serve_forever(self):
            calls.append(("serve",))

        def shutdown(self):
            calls.append(("shutdown",))

        def server_close(self):
            calls.append(("close",))

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            calls.append(("thread", target, name, daemon))

        def start(self):
            calls.append(("start",))

    adapter = _isolated_adapter(monkeypatch, tmp_path, extra)
    monkeypatch.setattr(adapter_module, "_A2AServer", FakeServer)
    monkeypatch.setattr(adapter_module.threading, "Thread", FakeThread)

    assert asyncio.run(adapter.connect()) is True
    assert adapter.is_connected is True
    assert calls[0][0:2] == ("server", ("127.0.0.1", 9900))
    assert [call[2] for call in calls if call[0] == "thread"] == [
        "a2a-http",
        "a2a-watchdog",
    ]

    asyncio.run(adapter.disconnect())
    assert ("shutdown",) in calls
    assert ("close",) in calls
