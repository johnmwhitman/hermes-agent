"""Tests for the gateway control socket (#92091 migration step 1)."""

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

import gateway.control_socket as control_socket

from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    identify_gateway,
    query_gateway_control,
    resolve_client_socket_path,
    resolve_server_socket_path,
    windows_pipe_name,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-socket transport; the named-pipe half is covered on the wine2e lane",
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def _serve(home: Path, handlers=None):
    """Context helper: start a server in a fresh loop, yield inside coro."""
    return GatewayControlServer(home, verb_handlers=handlers)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_short_home_binds_in_home(tmp_path: Path):
    # A home short enough for sun_path binds in-home with no pointer.
    # tmp_path can exceed the limit on CI runners, so build one in the
    # system temp root directly.
    import tempfile

    try:
        short_root = Path(tempfile.mkdtemp(prefix="hgw-", dir="/tmp"))
    except OSError:
        pytest.skip("/tmp not writable on this host")
    try:
        short_home = short_root / ".hermes"
        short_home.mkdir()
        assert len(str(short_home / "gateway.sock").encode()) <= 100
        bind, pointer = resolve_server_socket_path(short_home)
        assert bind == short_home / "gateway.sock"
        assert pointer is None
    finally:
        import shutil

        shutil.rmtree(short_root, ignore_errors=True)


def test_long_home_uses_pointer_fallback(tmp_path: Path):
    deep = tmp_path / ("x" * 120) / ".hermes"
    deep.mkdir(parents=True)
    bind, pointer = resolve_server_socket_path(deep)
    assert bind != deep / "gateway.sock"
    assert len(str(bind).encode()) <= 100
    assert pointer == deep / "gateway.sock.path"


def test_client_resolution_prefers_direct_then_pointer(home: Path, tmp_path: Path):
    assert resolve_client_socket_path(home) is None
    # pointer file to an existing socket-ish file
    target = tmp_path / "elsewhere.sock"
    target.touch()
    (home / "gateway.sock.path").write_text(str(target))
    assert resolve_client_socket_path(home) == target
    # direct file wins over pointer
    direct = home / "gateway.sock"
    direct.touch()
    assert resolve_client_socket_path(home) == direct


def test_windows_pipe_name_is_stable_and_home_scoped(tmp_path: Path):
    a = windows_pipe_name(tmp_path / "a")
    b = windows_pipe_name(tmp_path / "b")
    assert a.startswith(r"\\.\pipe\hermes-gateway-")
    assert a != b
    assert a == windows_pipe_name(tmp_path / "a")


# ---------------------------------------------------------------------------
# Server lifecycle + verbs (real sockets, real event loop)
# ---------------------------------------------------------------------------

def test_server_answers_identify_and_status(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home,
            verb_handlers={
                "identify": lambda: {"pid": 4242, "code_sha": "abc123", "protocol": 1},
                "status": lambda: {"gateway_state": "running"},
            },
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            ident = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "identify")
            )
            status = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "status")
            )
            return ident, status
        finally:
            await server.stop()

    ident, status = _run(scenario())
    assert ident == {"pid": 4242, "code_sha": "abc123", "protocol": 1}
    assert status == {"gateway_state": "running"}


def test_unknown_verb_and_malformed_request(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            unknown = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "restart")
            )

            def raw_garbage():
                path = resolve_client_socket_path(home)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect(str(path))
                    s.sendall(b"this is not json\n")
                    return s.recv(65536)

            garbage_reply = await loop.run_in_executor(None, raw_garbage)
            return unknown, garbage_reply
        finally:
            await server.stop()

    unknown, garbage_reply = _run(scenario())
    # unknown verb → ok:false → client returns None (fallback signal)
    assert unknown is None
    payload = json.loads(garbage_reply.decode())
    assert payload["ok"] is False
    assert payload["protocol"] == CONTROL_PROTOCOL_VERSION


def test_stop_removes_socket_and_pointer(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        bind, _ = resolve_server_socket_path(home)
        assert bind.exists()
        await server.stop()
        return bind

    bind = _run(scenario())
    assert not bind.exists()
    assert resolve_client_socket_path(home) is None
    # queries after stop cleanly return None
    assert query_gateway_control(home, "identify") is None


def test_stale_socket_file_is_replaced_on_bind(home: Path):
    # Plant the stale file at wherever the server will actually bind
    # (in-home OR the temp-dir fallback, depending on path length).
    bind, _ = resolve_server_socket_path(home)
    bind.parent.mkdir(parents=True, exist_ok=True)
    bind.touch()  # crashed predecessor's leftover

    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 7}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 7}


def test_long_home_end_to_end_via_pointer(tmp_path: Path):
    deep = tmp_path / ("p" * 120) / ".hermes"
    deep.mkdir(parents=True)

    async def scenario():
        server = GatewayControlServer(
            deep, verb_handlers={"identify": lambda: {"pid": 9}}
        )
        assert await server.start()
        try:
            assert (deep / "gateway.sock.path").is_file()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(deep))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 9}
    assert not (deep / "gateway.sock.path").exists()


def test_no_socket_returns_none_fast(home: Path):
    assert identify_gateway(home) is None
    assert query_gateway_control(home, "status") is None


@pytest.mark.parametrize(
    "response",
    [
        {"ok": True, "id": 1, "protocol": 2, "result": {"pid": 1}},
        {"ok": True, "id": 2, "protocol": 1, "result": {"pid": 1}},
        {"ok": True, "protocol": 1, "result": {"pid": 1}},
        {"ok": True, "id": 1, "result": {"pid": 1}},
    ],
)
def test_strict_query_rejects_mismatched_protocol_or_request_id(
    home: Path, monkeypatch, response
):
    monkeypatch.setattr(
        control_socket,
        "_query_unix_socket",
        lambda *_args, **_kwargs: json.dumps(response).encode() + b"\n",
    )

    with pytest.raises(RuntimeError, match="protocol|request"):
        query_gateway_control(home, "identify", strict=True)


def _strict_identity(home: Path, *, pid: int = 321, start_time: float = 10.0):
    return {
        "protocol": CONTROL_PROTOCOL_VERSION,
        "pid": pid,
        "start_time": start_time,
        "kind": "hermes-gateway",
        "hermes_home": str(home.resolve()),
    }


def test_strict_identity_rejects_absent_pid(home: Path, monkeypatch):
    identity = _strict_identity(home)
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda *_a, **_k: identity)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)

    with pytest.raises(RuntimeError, match="not live"):
        identify_gateway(home, strict=True)


def test_strict_identity_rejects_recycled_pid(home: Path, monkeypatch):
    identity = _strict_identity(home, start_time=10.0)
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda *_a, **_k: identity)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr("gateway.status._get_process_start_time", lambda _pid: 20.0)

    with pytest.raises(RuntimeError, match="identity changed"):
        identify_gateway(home, strict=True)


def test_strict_identity_rejects_misbound_home(home: Path, tmp_path: Path, monkeypatch):
    identity = _strict_identity(tmp_path / "other-home")
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda *_a, **_k: identity)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr("gateway.status._get_process_start_time", lambda _pid: 10.0)

    with pytest.raises(RuntimeError, match="home|profile"):
        identify_gateway(home, strict=True)


def test_strict_identity_accepts_explicit_multiplex_profile_binding(
    tmp_path: Path, monkeypatch
):
    default_home = tmp_path / ".hermes"
    profile_home = default_home / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    identity = _strict_identity(default_home)
    identity["served_profiles"] = ["default", "coder"]
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda *_a, **_k: identity)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr("gateway.status._get_process_start_time", lambda _pid: 10.0)

    assert identify_gateway(profile_home, strict=True) == identity


def test_strict_identity_rejects_same_profile_slug_from_other_root(
    tmp_path: Path, monkeypatch
):
    expected_home = tmp_path / "tenant-a" / "profiles" / "coder"
    expected_home.mkdir(parents=True)
    other_root = tmp_path / "tenant-b"
    identity = _strict_identity(other_root)
    identity["served_profiles"] = ["default", "coder"]
    monkeypatch.setattr(control_socket, "query_gateway_control", lambda *_a, **_k: identity)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr("gateway.status._get_process_start_time", lambda _pid: 10.0)

    with pytest.raises(RuntimeError, match="home|profile"):
        identify_gateway(expected_home, strict=True)


def test_unix_strict_query_enforces_overall_slow_drip_deadline(
    home: Path, monkeypatch
):
    class SlowDripSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendall(self, _request):
            pass

        def recv(self, _size):
            time.sleep(0.01)
            return b"x"

    monkeypatch.setattr(
        control_socket, "resolve_client_socket_path", lambda *_a, **_k: home / "gateway.sock"
    )
    monkeypatch.setattr(control_socket.socket, "socket", lambda *_a, **_k: SlowDripSocket())

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline"):
        control_socket._query_unix_socket(home, b"{}\n", 0.025, strict=True)
    assert time.monotonic() - started < 0.25


def test_windows_strict_query_enforces_blocking_read_deadline(
    home: Path, monkeypatch
):
    released = threading.Event()

    class BlockingPipe:
        def write(self, _request):
            return None

        def read(self, _size):
            released.wait(timeout=1.0)
            return b""

        def close(self):
            released.set()

    monkeypatch.setattr("builtins.open", lambda *_a, **_k: BlockingPipe())

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline"):
        control_socket._query_windows_pipe(home, b"{}\n", 0.025, strict=True)
    assert time.monotonic() - started < 0.25


def test_default_identify_payload_shape(home: Path, monkeypatch):
    """The real identify handler carries the fleet-consumer contract fields."""
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def scenario():
        server = GatewayControlServer(home)  # default handlers
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    ident = _run(scenario())
    assert ident is not None
    assert ident["protocol"] == CONTROL_PROTOCOL_VERSION
    assert ident["pid"] == __import__("os").getpid()
    # contract keys exist even when values are None/absent-degradable
    for key in ("hermes_home", "supervisor", "kind", "start_time"):
        assert key in ident
    assert ident["supervisor"] in {
        "systemd",
        "launchd",
        "desktop",
        "external",
        "manual",
    }


# ---------------------------------------------------------------------------
# Consumer integration: fleet matrix + inventory prefer socket, fall back
# ---------------------------------------------------------------------------

def _fake_identity(pid: int, sha: str):
    return {
        "protocol": 1,
        "pid": pid,
        "code_sha": sha,
        "code_version": "9.9.9",
        "supervisor": "systemd",
        "kind": "hermes-gateway",
    }


def test_collect_fleet_versions_prefers_socket(tmp_path: Path, monkeypatch):
    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    # stale state file that would report a WRONG pid — socket must win
    (home / "gateway_state.json").write_text(
        json.dumps({"pid": 1, "code_sha": "stalefile", "kind": "hermes-gateway"})
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(31337, "HEADSHA"),
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    entry = fleet[0]
    assert entry["pid"] == 31337
    assert entry["state"] == "current"
    assert entry["source"] == "socket"


def test_collect_fleet_versions_falls_back_to_state_file(tmp_path: Path, monkeypatch):
    import os

    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway", lambda h, **kw: None
    )
    (home / "gateway_state.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # a live pid so _pid_exists passes
                "code_sha": "OLDSHA",
                "kind": "hermes-gateway",
            }
        )
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    assert fleet[0]["pid"] == os.getpid()
    assert fleet[0]["state"] == "stale"
    assert "source" not in fleet[0]


def test_runtime_inventory_dedupes_same_pid_across_homes(tmp_path: Path, monkeypatch):
    """One multiplex gateway answering identify for two profile homes must
    yield exactly ONE runtime record (reviewer point on #92447)."""
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()
    profiles_root = tmp_path / "profiles"
    (profiles_root / "coder").mkdir(parents=True)

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(777, "SHA777"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1, [r.__dict__ for r in gws]
    assert gws[0].pid == 777


def test_runtime_inventory_prefers_socket_supervisor(tmp_path: Path, monkeypatch):
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(555, "SHA555"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1
    assert gws[0].pid == 555
    # supervisor comes from the gateway's own declaration, not a PID scan
    assert gws[0].supervisor == "systemd"
    assert gws[0].code_sha == "SHA555"
