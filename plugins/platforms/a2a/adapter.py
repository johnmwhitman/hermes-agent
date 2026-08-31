"""
A2A inbound platform adapter — exposes Hermes as an A2A-discoverable agent.

Design (the #11025 insight, done as a plugin with zero core edits):
  - Runs a stdlib http.server in a daemon thread (no a2a-sdk, no asyncio loop
    dependency at register() time — avoids the a2a_fleet "register outside a
    loop" bug class).
  - Serves the A2A v1.0 Agent Card at GET /.well-known/agent-card.json (and legacy agent.json).
  - JSON-RPC at POST /: message/send, message/stream (SSE), tasks/get,
    tasks/list, tasks/cancel, tasks/subscribe, tasks/pushNotificationConfig/create,
    tasks/pushNotificationConfig/get, tasks/pushNotificationConfig/list,
    tasks/pushNotificationConfig/delete.
  - Push notifications: config accepted inline in message/send
    (configuration.taskPushNotificationConfig) or via the create method;
    payloads are v1.0 StreamResponse objects, HMAC-signed.
  - Metrics at GET /metrics.
  - Each inbound task is filtered + framed (security.wrap_inbound) and routed
    into the agent's LIVE gateway session via the normal MessageEvent path, so
    the agent that replies is the same one talking to its user — full memory
    and context, not a throwaway clone.
  - The agent's reply comes back through ``adapter.send()``; we override that to
    fulfil a per-task Future the HTTP handler is blocked on, turning the
    async gateway into a synchronous request/response for the A2A caller.
    ``on_processing_complete`` resolves failures/cancellations promptly.
  - Every exchange is persisted to disk and audit-logged.

Bind safety: with no token configured, the server binds 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.config import Platform

from . import _listener_policy, posture, protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9900
_ORPHAN_TIMEOUT = 300  # floor: seconds before a pending task is considered orphaned
_ORPHAN_GRACE = 60  # slack added on top of a served agent's configured route timeout
_WATCHDOG_INTERVAL = 60  # seconds between orphaned task watchdog runs
_MAX_BODY = 1_048_576  # 1MB max request body — prevents DoS via memory exhaustion
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments


@dataclass(frozen=True)
class _IngressCapability:
    nonce: object
    snapshot: tuple[Any, ...]


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


def _truthy(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _string_list(value: Any) -> list[str]:
    """Normalize list-or-comma-string route policy values without coercion."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _default_agent_name() -> str:
    name = os.getenv("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


def _clean_slug(value: str) -> str:
    """Return a URL-safe-ish single-segment slug for a served agent."""
    slug = str(value or "").strip().strip("/")
    return "" if slug in ("", "root") else slug.split("/")[0]


def _join_url(base: str, prefix: str) -> str:
    base = (base or "").strip() or "/"
    if not base.endswith("/"):
        base += "/"
    prefix = (prefix or "").strip("/")
    if not prefix:
        return base
    return urllib.parse.urljoin(base, prefix + "/")


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.getenv("HERMES_PROFILE", "default") or "default"


def _profile_home(profile: str) -> Optional[str]:
    try:
        from hermes_cli.profiles import get_profile_dir
        return str(get_profile_dir(profile))
    except Exception:
        if not profile or profile == "default":
            try:
                from hermes_cli.config import get_hermes_home
                return str(get_hermes_home())
            except Exception:
                return None
        return os.path.expanduser(f"~/.hermes/profiles/{profile}")


@contextmanager
def _served_profile_config_scope(agent: dict):
    """Temporarily apply only the served profile's configuration home."""
    profile = str(agent.get("profile") or "").strip() if isinstance(agent, dict) else ""
    home = _profile_home(profile)
    if not home:
        yield False
        return
    home_token = None
    try:
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_token = set_hermes_home_override(str(home))
    except Exception:
        if home_token is not None:
            reset_hermes_home_override(home_token)
        yield False
        return
    try:
        yield True
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)


@contextmanager
def _served_profile_secret_scope(agent: dict):
    """Apply secrets only after the served profile policy has validated."""
    home = _profile_home(
        str(agent.get("profile") or "").strip() if isinstance(agent, dict) else ""
    )
    if not home:
        yield False
        return
    secret_token = None
    with _served_profile_config_scope(agent) as scoped:
        if not scoped:
            yield False
            return
        try:
            from agent.secret_scope import (
                build_profile_secret_scope,
                reset_secret_scope,
                set_secret_scope,
            )
            from hermes_cli.env_loader import hydrate_profile_secret_sources

            hydrate_profile_secret_sources(Path(home))
            secret_token = set_secret_scope(build_profile_secret_scope(Path(home)))
        except Exception:
            yield False
            return
        try:
            yield True
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)


def _served_profile_toolset_scope(
    agent: dict,
) -> Optional[tuple[list[str], list[str]]]:
    """Resolve the exact toolset scope used by the destination agent.

    Forwarded served profiles execute through ``hermes chat`` and therefore
    use the CLI platform toolset law.  A route handled by the gateway's active
    profile executes locally as the A2A platform.  Read the served profile's
    raw config through the same managed-overlay and normalization steps as the
    gateway, then use the canonical platform resolver.  Missing, malformed, or
    unreadable profile config is not a request for every tool: callers must
    reject the task before assembling an unscoped catalog.
    """
    if not isinstance(agent, dict):
        return None
    profile = str(agent.get("profile") or "").strip()
    home = _profile_home(profile)
    if not home:
        return None
    config_path = Path(home) / "config.yaml"
    try:
        with _served_profile_config_scope(agent) as scoped:
            if not scoped:
                return None
            if config_path.is_file():
                from utils import fast_safe_load

                with config_path.open("r", encoding="utf-8") as handle:
                    raw_config = fast_safe_load(handle)
                if raw_config is not None and not isinstance(raw_config, dict):
                    return None
            elif agent.get("local") is True:
                # The active/default profile may legitimately have no user
                # config; canonical loading then supplies DEFAULT_CONFIG.
                pass
            else:
                return None

            # Forwarded profiles execute through ``hermes chat``. Resolve the
            # policy with that same canonical loader so DEFAULT_CONFIG,
            # environment expansion, normalization, and managed overlay stay
            # byte-for-byte semantic peers of destination construction.
            from hermes_cli.config import load_config

            config = load_config()
            if not isinstance(config, dict):
                return None

            from hermes_cli.tools_config import _get_platform_tools

            platform = "a2a" if agent.get("local") is True else "cli"
            platform_toolsets = config.get("platform_toolsets")
            if platform_toolsets is not None and not isinstance(platform_toolsets, dict):
                return None
            if (
                isinstance(platform_toolsets, dict)
                and platform in platform_toolsets
                and not isinstance(platform_toolsets[platform], list)
            ):
                return None
            coding_scope = None
            if platform == "cli":
                from agent.coding_context import coding_selection

                coding_scope = coding_selection(platform="cli", config=config)
                if coding_scope is not None and not isinstance(coding_scope, list):
                    return None
            enabled = sorted(
                coding_scope
                if coding_scope is not None
                else _get_platform_tools(config, platform)
            )
            agent_config = config.get("agent") or {}
            if not isinstance(agent_config, dict):
                return None
            raw_disabled = agent_config.get("disabled_toolsets")
            if raw_disabled is not None and not isinstance(
                raw_disabled, (str, list, tuple, set)
            ):
                return None
            from agent.skill_utils import parse_config_string_list

            disabled = [
                value.strip()
                for value in parse_config_string_list(raw_disabled or [])
                if value.strip()
            ]
            return enabled, disabled
    except Exception:
        logger.warning(
            "A2A: could not resolve served profile %r toolset scope",
            profile,
            exc_info=True,
        )
        return None


def _safe_context_slug(value: str, max_len: int = 96) -> str:
    """Create a bounded title slug with collision-resistant context identity."""
    raw = str(value or "")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._") or "ctx"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    prefix_len = max(1, max_len - len(digest) - 1)
    return f"{slug[:prefix_len]}-{digest}"[:max_len]


def _method_info(method: str) -> tuple[str, bool]:
    """Return (canonical_operation, is_v1_method).

    Canonical operation names are lowercase internal labels. v1 methods use the
    PascalCase names from A2A v1.0 §5.3/§9.4; legacy aliases remain accepted.
    """
    mapping = {
        "SendMessage": ("send", True),
        "message/send": ("send", False),
        "SendStreamingMessage": ("stream", True),
        "message/stream": ("stream", False),
        "GetTask": ("get", True),
        "tasks/get": ("get", False),
        "ListTasks": ("list", True),
        "tasks/list": ("list", False),
        "CancelTask": ("cancel", True),
        "tasks/cancel": ("cancel", False),
        "SubscribeToTask": ("subscribe", True),
        "tasks/subscribe": ("subscribe", False),
        "CreateTaskPushNotificationConfig": ("push_create", True),
        "tasks/pushNotificationConfig/create": ("push_create", False),
        "tasks/pushNotificationConfig/set": ("push_create", False),
        "tasks/pushNotification/set": ("push_create", False),
        "GetTaskPushNotificationConfig": ("push_get", True),
        "tasks/pushNotificationConfig/get": ("push_get", False),
        "ListTaskPushNotificationConfigs": ("push_list", True),
        "tasks/pushNotificationConfig/list": ("push_list", False),
        "DeleteTaskPushNotificationConfig": ("push_delete", True),
        "tasks/pushNotificationConfig/delete": ("push_delete", False),
    }
    return mapping.get(method, ("", False))


class _A2AServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that carries a reference to its adapter."""

    daemon_threads = True

    def __init__(self, addr, handler_cls, adapter: "A2AAdapter"):
        super().__init__(addr, handler_cls)
        self.adapter = adapter


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the A2A JSON-RPC surface.

    Module-level (not a closure) so request routing is unit-testable; all
    state lives on ``self.server.adapter``.
    """

    @property
    def adapter(self) -> "A2AAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    # Silence the default stderr access log.
    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("A2A http: " + format, *args)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _request_public_url(self) -> str:
        """Derive the routable URL for this request.

        Priority: A2A_PUBLIC_URL env > X-Forwarded-Host / Host header (with
        scheme from X-Forwarded-Proto) > empty. Empty means "caller has no
        info, fall back to bind host". See gfdsa's k8s bind-host bug report
        (PR #41711).
        """
        explicit = os.getenv("A2A_PUBLIC_URL", "").strip()
        if explicit:
            return explicit
        host = self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")
        if not host:
            return ""
        host = host.split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/"

    def do_GET(self):  # noqa: N802
        route = self.adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            public_url = self._request_public_url() or None
            self._json(200, self.adapter._build_card(public_url, agent=agent))
            return
        if subpath in ("/", "/health"):
            payload = {
                "status": "ok",
                "agent": agent.get("name") or self.adapter.agent_name,
            }
            # Do not leak profile/tenant topology on remote unauthenticated GETs.
            # Agent Cards are intentionally public; health topology is not.
            if security.localhost_only() or security.authenticate(
                self.headers.get("Authorization"),
                self.client_address[0] if self.client_address else "",
            ) is not None:
                payload["served_agents"] = self.adapter._served_agent_summary(
                    public_url=self._request_public_url() or None)
            self._json(200, payload)
            return
        if subpath == "/metrics":
            self._json(200, protocol.metrics.snapshot())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        adapter = self.adapter
        client_ip = self.client_address[0] if self.client_address else ""

        # Identity comes from the presented credential (or the socket in
        # localhost-only mode) — never from the request body.
        identity = security.authenticate(self.headers.get("Authorization"), client_ip)
        if identity is None:
            self._json(401, protocol.jsonrpc_error(None, protocol.ERR_UNAUTHORIZED, "unauthorized"))
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                self._json(413, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "payload too large"))
                return
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "parse error"))
            return

        if not isinstance(req, dict):
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object"))
            return

        req_id = req.get("id")
        method = str(req.get("method", ""))
        params = req.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, "params must be an object"))
            return

        version = (self.headers.get("A2A-Version") or "").strip()
        if version and version not in {"1.0", "1.0.0"}:
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"))
            return

        operation, is_v1 = _method_info(method)
        route = adapter._route_for_request(self.path, params)
        if route.get("error"):
            self._json(400, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, route["error"]))
            return
        agent = route["agent"]

        if not security.is_authorized_for_agent(identity, agent):
            self._json(403, protocol.jsonrpc_error(
                req_id, protocol.ERR_UNTRUSTED_PEER,
                f"peer '{identity}' is not authorized for this served agent",
            ))
            return

        if not adapter._rate_limiter.allow(identity):
            protocol.metrics.rate_limit_triggers += 1
            self._json(429, protocol.jsonrpc_error(req_id, protocol.ERR_RATE_LIMITED, "rate limit exceeded"))
            return

        if not security.is_trusted_peer(identity):
            self._json(403, protocol.jsonrpc_error(
                req_id, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"))
            return

        if not operation:
            self._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"))
            return

        if operation == "send":
            self._json(200, adapter._rpc_message_send(
                req_id, params, identity, agent=agent, v1_response=is_v1,
                credential_authenticated=not security.localhost_only(),
            ))
            return
        if operation == "stream":
            adapter._rpc_message_stream(
                self, req_id, params, identity, agent=agent,
                credential_authenticated=not security.localhost_only(),
            )
            return
        if operation == "get":
            self._json(200, adapter._rpc_tasks_get(req_id, params, agent=agent, peer=identity))
            return
        if operation == "list":
            self._json(200, adapter._rpc_tasks_list(req_id, params, agent=agent, peer=identity))
            return
        if operation == "cancel":
            self._json(200, adapter._rpc_tasks_cancel(req_id, params, agent=agent, peer=identity))
            return
        if operation == "subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent, peer=identity)
            return
        if operation == "push_create":
            self._json(200, adapter._rpc_push_config_create(req_id, params, agent=agent, peer=identity))
            return
        if operation == "push_get":
            self._json(200, adapter._rpc_push_config_get(req_id, params, agent=agent, peer=identity))
            return
        if operation == "push_list":
            self._json(200, adapter._rpc_push_config_list(req_id, params, agent=agent, peer=identity))
            return
        if operation == "push_delete":
            self._json(200, adapter._rpc_push_config_delete(req_id, params, agent=agent, peer=identity))
            return



class A2AAdapter(BasePlatformAdapter):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        platform = Platform("a2a")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self._inbound_enabled, self._listener_mode_error = _listener_policy(extra)
        # Outbound-only profiles never consume bind configuration.  In
        # particular, a stale inherited A2A_PORT must not break a remote client
        # profile that will not construct a listener.
        self.port = 0
        self.host = ""
        if self._inbound_enabled and self._listener_mode_error is None:
            self.port = int(os.getenv("A2A_PORT") or extra.get("port", _DEFAULT_PORT))
            self.host = security.resolve_bind_host()
        self.agent_name = _default_agent_name()
        self._advertised_toolsets = [
            t.strip() for t in (
                list(extra.get("advertised_toolsets") or [])
                or os.getenv("A2A_ADVERTISED_TOOLSETS", "").split(",")
            ) if str(t).strip()
        ]
        self._active_profile = _active_profile_name()
        self._agents = self._load_served_agents(extra)
        # Hollow-reply guard: a forwarded profile that answers without a single
        # tool call has no tool-backed evidence for its claims (a zero-tool
        # worker fabricated OS error strings and was recorded a2a_complete).
        # Default on; ``platforms.a2a.extra.require_tools: false`` disables.
        self._require_tools = _truthy(extra.get("require_tools", True))

        self._httpd: Optional[_A2AServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-adapter protocol state (not module-global): task store, anti-loop
        # turn tracking, and rate limiting.
        self.tasks = protocol.TaskStore()
        self._turns = protocol.TurnTracker()
        self._rate_limiter = protocol.RateLimiter()

        # Forwarded profile sessions: map (profile, agent_slug, context_id) -> session_id.
        self._profile_sessions: Dict[tuple[str, str, str], str] = {}
        # First-contact discovery queries the profile's shared state.db for the
        # newest source=a2a row. Serialize all forwards to one profile so two
        # contexts cannot claim each other's newly created session.
        self._profile_session_locks: Dict[str, threading.Lock] = {}
        self._profile_session_locks_guard = threading.Lock()
        # Durable-in-process posture bindings for A2A context resumes.  The
        # receiving gateway also persists the same binding on SessionSource;
        # this map covers the adapter's direct task path.
        self._posture_bindings: Dict[tuple[str, str], dict] = posture.load_persisted_bindings()
        self._posture_lock = threading.Lock()
        self._a2a_ingress_nonce = object()

        # Pending reply futures, keyed by task_id. Each future resolves to a
        # (state, text) tuple. _pending_order keeps per-context FIFO order so
        # adapter.send() — which only knows the context — resolves the oldest
        # outstanding task for that context (no cross-talk between concurrent
        # requests sharing a context).
        self._pending: Dict[str, tuple[tuple[str, str, str], Future]] = {}
        self._pending_order: Dict[tuple[str, str, str], deque[str]] = {}
        self._pending_lock = threading.Lock()

        # Orphaned task watchdog
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

    @staticmethod
    def _ingress_snapshot(source: Any) -> tuple[Any, ...] | None:
        """Snapshot every field that contributes to an A2A posture decision."""
        binding = posture.PostureBinding.from_value(
            getattr(source, "a2a_binding", None)
        )
        if binding is None:
            return None
        return (
            getattr(source, "a2a_mutation_requested", None),
            getattr(source, "a2a_mutation_enabled", None),
            getattr(source, "a2a_credential_authenticated", None),
            getattr(source, "a2a_peer_trusted", None),
            str(getattr(source, "a2a_peer", "")),
            str(getattr(source, "a2a_agent_slug", "")),
            str(getattr(source, "a2a_context_id", "")),
            str(getattr(source, "chat_id", "")),
            str(getattr(source, "profile", "")),
            tuple(sorted(str(v) for v in (getattr(source, "a2a_mutable_toolsets", ()) or ()))),
            tuple(sorted(str(v) for v in (getattr(source, "a2a_allowed_tool_names", ()) or ()))),
            str(getattr(source, "a2a_toolset_fingerprint", "")),
            tuple(sorted(binding.to_dict().items())),
        )

    def _stamp_ingress_source(self, source: Any):
        """Mint one adapter-private, nonserializable ingress capability."""
        snapshot = self._ingress_snapshot(source)
        if snapshot is None:
            raise ValueError("cannot stamp malformed A2A posture binding")
        capability = _IngressCapability(self._a2a_ingress_nonce, snapshot)
        return replace(source, _a2a_ingress_capability=capability)

    def _validate_ingress_source(self, source: Any) -> bool:
        """Validate source ownership and its immutable adapter snapshot."""
        adapter_ref = getattr(source, "_transport_adapter_ref", None)
        owner = adapter_ref() if callable(adapter_ref) else None
        capability = getattr(source, "_a2a_ingress_capability", None)
        snapshot = self._ingress_snapshot(source)
        return bool(
            owner is self
            and isinstance(capability, _IngressCapability)
            and capability.nonce is self._a2a_ingress_nonce
            and snapshot is not None
            and capability.snapshot == snapshot
            and posture.source_binding_valid(source)
        )

    def _rebind_readonly_source(self, source: Any):
        """Narrow a valid live source and mint a replacement capability."""
        updates = posture.readonly_source_updates(source)
        if updates is None:
            return None
        return self._stamp_ingress_source(replace(source, **updates))

    @property
    def name(self) -> str:
        return "A2A"

    @property
    def authorization_is_upstream(self) -> bool:
        """A2A authenticates every inbound request via bearer token (or
        localhost-only binding) in ``do_POST`` before dispatch — the identity
        is already authorized upstream. Without this override, the gateway's
        per-platform user allow-list (``{PLATFORM}_ALLOWED_USERS``) rejects
        A2A peers because their identity is a token-derived name or pod IP,
        not a platform account the operator configures in an env allow-list.

        This is authorization delegated to the A2A bearer-token transport,
        not a fail-open: every request is 401'd if the credential is wrong.
        Reported by kuangmi-bit (PR #41711 comment, Jun 27).
        """
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self, **_kwargs) -> bool:
        # Gateway reconnection plumbing passes adapter-agnostic kwargs such as
        # ``is_reconnect``. A2A does not need them, but accepting them keeps the
        # plugin compatible with the BasePlatformAdapter lifecycle contract.
        if self._listener_mode_error is not None:
            logger.error(
                "A2A: invalid listener configuration: %s",
                self._listener_mode_error,
            )
            self._set_fatal_error(
                "invalid_listener_mode",
                self._listener_mode_error,
                retryable=False,
            )
            return False

        if not self._inbound_enabled:
            self._mark_connected()
            logger.info(
                "A2A: outbound client mode enabled; inbound HTTP listener disabled"
            )
            return True

        # Capture the running gateway loop so the HTTP thread can marshal
        # events onto it via run_coroutine_threadsafe.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        try:
            self._httpd = _A2AServer((self.host, self.port), A2ARequestHandler, self)
        except OSError as e:
            logger.error("A2A: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"A2A bind failed: {e}", retryable=True)
            return False

        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="a2a-http",
            daemon=True,
        )
        self._server_thread.start()

        # Reset watchdog state for reconnection (disconnect sets the event)
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="a2a-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

        self._mark_connected()

        exposure = "localhost-only" if security.localhost_only() else "REMOTE (bearer auth)"
        logger.info(
            "A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)",
            self.host, self.port, exposure, self.agent_name, len(self._agents),
        )
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        # Fail any in-flight replies so blocked HTTP threads don't hang.
        with self._pending_lock:
            for _ctx, fut in self._pending.values():
                if not fut.done():
                    fut.set_result((protocol.STATE_FAILED, "[agent shutting down]"))
            self._pending.clear()
            self._pending_order.clear()

    # ── Orphaned task watchdog ─────────────────────────────────────────────

    def _orphan_timeout_for(self, agent_slug: str) -> int:
        """Orphan deadline for a task routed to ``agent_slug``.

        A forwarded profile may legitimately run for its configured route
        ``timeout`` (e.g. 900 s); the watchdog must not fail the task at the
        300 s floor while the subprocess is still allowed to run.
        """
        agent = self._agents.get(agent_slug or "") or {}
        try:
            route_timeout = int(agent.get("timeout") or 0)
        except (TypeError, ValueError):
            route_timeout = 0
        return max(_ORPHAN_TIMEOUT, route_timeout + _ORPHAN_GRACE)

    def _watchdog_loop(self) -> None:
        """Background thread that fails orphaned tasks (keeps them queryable)."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                for tid in self.tasks.fail_orphans(
                    _ORPHAN_TIMEOUT,
                    timeout_for=lambda rec: self._orphan_timeout_for(rec.get("agent_slug", "")),
                ):
                    logger.warning("A2A: orphaned task %s marked failed", tid)
                    protocol.metrics.tasks_failed += 1
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    # ── Agent routing + Agent Cards ───────────────────────────────────────

    def _load_global_a2a_config(self) -> dict:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _load_served_agents(self, extra: dict) -> dict[str, dict]:
        """Load served-agent routing config.

        Preferred config location is ``platforms.a2a.extra.agents``. A top-level
        ``a2a_served_agents`` fallback is accepted for scripts/tests. Root/default
        always maps to the live gateway session for backward compatibility.
        """
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            cfg = self._load_global_a2a_config()
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")

        agents: dict[str, dict] = {}
        default_desc = os.getenv(
            "A2A_AGENT_DESCRIPTION",
            "Hermes Agent — a general-purpose agent reachable over A2A.",
        )
        agents[""] = {
            "slug": "",
            "path": "",
            "tenant": "",
            "profile": self._active_profile,
            "local": True,
            "name": self.agent_name,
            "description": default_desc,
            "advertised_toolsets": self._advertised_toolsets,
            "advertise_mutable_capability": extra.get("advertise_mutable_capability") is True,
            "mutable_toolsets": [
                str(t).strip() for t in (extra.get("mutable_toolsets") or [])
                if str(t).strip()
            ],
            "mutation_allowed_peers": _string_list(extra.get("mutation_allowed_peers")),
            "allowed_peers": _string_list(extra.get("allowed_peers")),
            "mutable_tool_names": [
                str(t).strip() for t in (extra.get("mutable_tool_names") or [])
                if str(t).strip()
            ],
        }

        reserved = {"health", "metrics", ".well-known"}
        tenants: dict[str, str] = {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw or []) if isinstance(raw, list) else []
        for key, val in items:
            if not isinstance(val, dict):
                continue
            slug = _clean_slug(str(val.get("slug") or val.get("id") or key))
            if not slug:
                continue
            path_segment = _clean_slug(str(val.get("path") or slug))
            if not path_segment or path_segment in reserved:
                logger.warning("A2A: ignoring served agent %r with reserved/invalid path %r", slug, path_segment)
                continue
            profile = str(val.get("profile") or slug).strip()
            path = "/" + path_segment
            toolsets = val.get("advertised_toolsets") or val.get("toolsets") or []
            if isinstance(toolsets, str):
                toolsets = [t.strip() for t in toolsets.split(",") if t.strip()]
            capabilities = val.get("capabilities") or []
            if isinstance(capabilities, str):
                capabilities = [t.strip() for t in capabilities.split(",") if t.strip()]
            if isinstance(val.get("local"), bool):
                local = val["local"]
            else:
                local = profile in ("", self._active_profile)
            model = str(val.get("model") or "").strip()
            provider = str(val.get("provider") or "").strip()
            # Any explicit pin overrides the profile default — surface it.
            # subs/* shims are text-only; pinning one resurrects the 08-15
            # hollow-reply class (fabricated OS errors, zero tool rows).
            if model:
                logger.warning(
                    "A2A: served agent %r pins model %r — forwarded tasks override the profile default",
                    slug, model,
                )
                model_l = model.lower()
                if model_l.startswith("subs/") or "/subs/" in model_l:
                    logger.warning(
                        "A2A: refusing served agent %r — model %r is a text-only subs/* shim",
                        slug, model,
                    )
                    continue
            tenant = str(val.get("tenant") or slug).strip()
            if tenant:
                if tenant in tenants:
                    logger.warning(
                        "A2A: ignoring served agent %r with duplicate tenant %r already used by %r",
                        slug, tenant, tenants[tenant],
                    )
                    continue
                tenants[tenant] = slug
            agents[slug] = {
                "slug": slug,
                "path": path,
                "tenant": tenant,
                "profile": profile or slug,
                "local": local,
                "name": str(val.get("name") or f"Hermes {slug}"),
                "description": str(val.get("description") or f"Hermes profile '{profile or slug}' exposed over A2A."),
                "advertised_toolsets": list(toolsets or []),
                # Descriptive product capabilities are not executable toolset
                # claims. Keep them separate so registry failure can preserve
                # labels such as "research" without fail-opening a concrete
                # mutable toolset such as "terminal".
                "advertised_capabilities": list(capabilities or []),
                "advertise_mutable_capability": val.get("advertise_mutable_capability") is True,
                "mutable_toolsets": [
                    str(t).strip() for t in (
                        val.get("mutable_toolsets")
                        or val.get("mutable_toolsets_for_a2a")
                        or []
                    ) if str(t).strip()
                ],
                "mutation_allowed_peers": _string_list(
                    val.get("mutation_allowed_peers")
                    or val.get("allowed_mutation_peers")
                ),
                "allowed_peers": _string_list(
                    val.get("allowed_peers") or val.get("authorized_peers")
                ),
                "mutable_tool_names": [
                    str(t).strip() for t in (val.get("mutable_tool_names") or [])
                    if str(t).strip()
                ],
                "timeout": int(val.get("timeout") or _reply_timeout()),
                "model": model,
                "provider": provider,
            }
        return agents

    def _served_agent_summary(self, public_url: Optional[str] = None) -> list[dict]:
        base = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        return [
            {
                "slug": a["slug"] or "default",
                "name": a.get("name"),
                "url": _join_url(base, a.get("path", "")),
                "tenant": a.get("tenant") or None,
                "profile": a.get("profile"),
                "local": bool(a.get("local")),
            }
            for a in self._agents.values()
        ]

    def _route_for_path(self, raw_path: str) -> dict:
        path = urllib.parse.urlsplit(raw_path or "/").path or "/"
        # Longest prefix wins. Default/root agent is the fallback.
        for agent in sorted(self._agents.values(), key=lambda a: len(a.get("path", "")), reverse=True):
            prefix = agent.get("path", "") or ""
            if prefix and (path == prefix or path.startswith(prefix + "/")):
                subpath = path[len(prefix):] or "/"
                if not subpath.startswith("/"):
                    subpath = "/" + subpath
                return {"agent": agent, "subpath": subpath}
        return {"agent": self._agents[""], "subpath": path}

    def _route_for_request(self, raw_path: str, params: dict) -> dict:
        route = self._route_for_path(raw_path)
        agent = route["agent"]
        tenant = str((params or {}).get("tenant") or "")
        # If no URL prefix chose a non-default agent, allow v1.0 tenant routing.
        if agent.get("slug") == "" and tenant:
            matches = [a for a in self._agents.values() if a.get("tenant") == tenant]
            if matches:
                route = {"agent": matches[0], "subpath": route["subpath"]}
                agent = matches[0]
        expected = str(agent.get("tenant") or "")
        if tenant and expected and tenant != expected:
            return {"error": f"tenant {tenant!r} does not match routed agent {agent.get('slug') or 'default'}"}
        return route

    def _build_card(self, public_url: Optional[str] = None, agent: Optional[dict] = None) -> dict:
        # Prefer per-request public URL (from X-Forwarded-Host / Host /
        # A2A_PUBLIC_URL) over bind host, so peers can call back when we're
        # behind a reverse proxy.
        if agent is None:
            agent = self._agents[""]
        if (
            not isinstance(agent, dict)
            or "slug" not in agent
            or not isinstance(agent.get("slug"), str)
        ):
            raise ValueError("invalid served A2A agent route")
        base = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        url = _join_url(base, agent.get("path", ""))
        return protocol.build_agent_card(
            name=agent.get("name") or self.agent_name,
            url=url,
            description=agent.get("description") or "Hermes Agent — a general-purpose agent reachable over A2A.",
            skills=self._advertised_skills(agent),
            streaming=bool(agent.get("local", True)),
            push_notifications=True,
            auth_required=not security.localhost_only(),
            tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
        """Dynamic Agent Card skills from the live tool registry.

        The card reflects what the agent can actually do right now. An
        explicit ``advertised_toolsets`` config (or A2A_ADVERTISED_TOOLSETS)
        restricts what we advertise; without a registry we fall back to that
        static list.
        """
        try:
            from tools.registry import registry as tool_registry
            names = tool_registry.get_registered_toolset_names()
            configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
            descriptive = (agent or {}).get("advertised_capabilities") or []
            # A route that declares only descriptive capabilities is still a
            # restricted card. Do not interpret its empty executable-toolset
            # list as permission to inherit every registry capability.
            allowed = set(configured or descriptive or []) or None
            mapping = {
                n: tool_registry.get_tool_names_for_toolset(n)
                for n in names
                if allowed is None or n in allowed
            }
            # Agent Cards are public capability declarations.  Keep the
            # default surface closed; mutable capabilities require a route
            # explicitly opting into their authenticated advertisement.
            if not bool((agent or {}).get("advertise_mutable_capability")):
                mapping = {
                    key: readonly_values
                    for key, values in mapping.items()
                    if (readonly_values := [
                        name for name in values
                        if name in posture.READONLY_TOOL_NAMES
                    ])
                }
            if mapping:
                skills = protocol.skills_from_toolsets(mapping)
                existing = {skill.get("name") for skill in skills}
                remaining = [
                    name for name in descriptive if str(name) not in existing
                ]
                if remaining:
                    skills.extend(protocol.skills_from_toolsets(remaining))
                return skills
            if descriptive:
                return protocol.skills_from_toolsets(descriptive)
            # Registry discovery succeeded, but every selected concrete
            # capability was filtered out. An empty public card is more honest
            # than inventing the helper's generic fallback skill.
            return []
        except Exception:
            logger.debug("A2A: tool registry unavailable for Agent Card", exc_info=True)
        configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
        # A route's static ``advertised_toolsets`` are an explicit capability
        # declaration. Without the registry we cannot expand and classify
        # grouped toolsets, so the default card may advertise only names in
        # the closed read-only surface. Mutable fallback advertising requires
        # the same explicit route opt-in as the normal registry path.
        if not bool((agent or {}).get("advertise_mutable_capability")):
            safe_configured = []
            try:
                from toolsets import get_toolset, resolve_toolset

                for name in configured or []:
                    label = str(name)
                    if label in posture.READONLY_TOOL_NAMES:
                        safe_configured.append(label)
                        continue
                    definition = get_toolset(label, include_registry=False)
                    if definition is None:
                        continue
                    resolved = set(resolve_toolset(label, include_registry=False))
                    if resolved and resolved.issubset(posture.READONLY_TOOL_NAMES):
                        safe_configured.append(label)
            except Exception:
                safe_configured = [
                    name for name in (configured or [])
                    if str(name) in posture.READONLY_TOOL_NAMES
                ]
            configured = safe_configured
        descriptive = (agent or {}).get("advertised_capabilities") or []
        return protocol.skills_from_toolsets(list(configured or []) + list(descriptive))

    # ── Pending reply plumbing ────────────────────────────────────────────

    def _add_pending(
        self, task_id: str, context_id: str, *, peer: str = "", agent_slug: str = "",
    ) -> Future:
        fut: Future = Future()
        scope = (str(peer), str(agent_slug), str(context_id))
        with self._pending_lock:
            self._pending[task_id] = (scope, fut)
            self._pending_order.setdefault(scope, deque()).append(task_id)
        return fut

    def _pop_pending(self, task_id: str) -> None:
        with self._pending_lock:
            entry = self._pending.pop(task_id, None)
            if entry:
                order = self._pending_order.get(entry[0])
                if order:
                    try:
                        order.remove(task_id)
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(entry[0], None)

    def _resolve_task(self, task_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            entry = self._pending.get(task_id)
            if entry and not entry[1].done():
                entry[1].set_result((state, text))
                return True
        return False

    def _resolve_oldest_for_scope(
        self, peer: str, agent_slug: str, context_id: str, state: str, text: str,
    ) -> bool:
        scope = (str(peer), str(agent_slug), str(context_id))
        with self._pending_lock:
            for task_id in self._pending_order.get(scope, ()):
                entry = self._pending.get(task_id)
                if entry and not entry[1].done():
                    entry[1].set_result((state, text))
                    return True
        return False

    def _scope_for_agent(self, agent: Optional[dict]) -> tuple[str, str]:
        if agent is None:
            agent = self._agents[""]
        if (
            not isinstance(agent, dict)
            or "slug" not in agent
            or not isinstance(agent.get("slug"), str)
        ):
            raise ValueError("invalid served A2A agent route")
        return str(agent.get("slug") or ""), str(agent.get("tenant") or "")

    def _forward_lock(self, profile: str) -> threading.Lock:
        with self._profile_session_locks_guard:
            lock = self._profile_session_locks.get(profile)
            if lock is None:
                lock = threading.Lock()
                self._profile_session_locks[profile] = lock
            return lock

    # ── Inbound task handling ─────────────────────────────────────────────

    def _prepare_task(
        self,
        params: dict,
        peer: str,
        agent: Optional[dict] = None,
        *,
        credential_authenticated: bool = False,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """Validate, register, and dispatch an inbound message.

        Returns (terminal_task, None) when the task ends immediately
        (rejected / not ready), else (None, pending) where pending carries
        the future the caller must wait on. Runs on an HTTP worker thread.
        """
        if agent is None:
            agent = self._agents[""]
        if (
            not isinstance(agent, dict)
            or "slug" not in agent
            or not isinstance(agent.get("slug"), str)
        ):
            return protocol.build_task(
                task_id=protocol.new_task_id(),
                context_id=protocol.extract_context_id(params) or protocol.new_context_id(),
                state=protocol.STATE_REJECTED,
                agent_text="invalid served A2A agent route",
            ), None
        if not security.is_trusted_peer(peer):
            return protocol.build_task(
                task_id=protocol.new_task_id(),
                context_id=protocol.extract_context_id(params) or protocol.new_context_id(),
                state=protocol.STATE_REJECTED,
                agent_text=f"peer '{peer}' is not globally trusted",
            ), None
        # Keep the route boundary inside the task primitive as well as the
        # HTTP handler. Direct/internal callers must not bypass a served
        # agent's peer allowlist or create any task/audit/session side effect.
        if not security.is_authorized_for_agent(peer, agent):
            return protocol.build_task(
                task_id=protocol.new_task_id(),
                context_id=protocol.extract_context_id(params) or protocol.new_context_id(),
                state=protocol.STATE_REJECTED,
                agent_text=f"peer '{peer}' is not authorized for this served agent",
            ), None
        request = posture.parse_mutation_request(params)
        context_from_peer = protocol.extract_context_id(params)
        context_id = context_from_peer or protocol.new_context_id()
        task_id = protocol.new_task_id()

        # This is intentionally before anti-loop tracking, audit, persistence,
        # and TaskStore.create: malformed/conflicting posture is rejected with
        # no observable bookkeeping side effect.
        if request.error:
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED, request.error,
            ), None

        mutable_toolsets = tuple(agent.get("mutable_toolsets") or ())
        slug = str(agent.get("slug") or "")
        allowed_peers = {
            str(value).strip()
            for value in (agent.get("mutation_allowed_peers") or ())
            if str(value).strip()
        }
        trusted = str(peer) in allowed_peers
        mutable = posture.effective_mutation(
            requested=request.requested,
            credential_authenticated=credential_authenticated,
            trusted_peer=trusted,
            mutable_toolsets=mutable_toolsets,
        )
        if request.requested and not mutable:
            # A mutation marker is an explicit request, so do not silently
            # downgrade an unauthorized caller into an apparently successful
            # read-only task.  Reject before anti-loop/audit/persistence.
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "mutation posture requires credential authentication, served-agent peer authorization, and explicit mutable toolsets.",
            ), None
        # A genuinely new context may establish a mutable binding. A context
        # with prior durable conversation/task evidence but no binding is a
        # restart/corruption case and must fall back read-only.
        # Resolve route toolsets through the same final Tool Search assembly
        # used by child construction, so the binding fingerprints directly
        # exposed tool names rather than raw deferred catalog entries or
        # policy labels such as ``filesystem``.
        toolset_scope = _served_profile_toolset_scope(agent)
        if toolset_scope is None:
            return protocol.build_task(
                task_id,
                context_id,
                protocol.STATE_REJECTED,
                "served profile tool scope is missing or malformed",
            ), None
        enabled_toolsets, disabled_toolsets = toolset_scope
        available_names = set(agent.get("mutable_tool_names") or ())
        try:
            from model_tools import get_tool_definitions
            with _served_profile_secret_scope(agent) as scoped:
                if not scoped:
                    raise RuntimeError("served profile home is unavailable")
                catalog = get_tool_definitions(
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    quiet_mode=True,
                    skip_tool_search_assembly=False,
                ) or []
            available_names.update(
                td.get("function", {}).get("name")
                for td in catalog
                if isinstance(td, dict) and isinstance(td.get("function"), dict)
            )
        except Exception:
            logger.warning(
                "A2A: scoped tool catalog assembly failed for served profile %r",
                agent.get("profile") if isinstance(agent, dict) else None,
                exc_info=True,
            )
            return protocol.build_task(
                task_id,
                context_id,
                protocol.STATE_REJECTED,
                "served profile tool catalog is unavailable",
            ), None
        mutable_names = posture.resolve_mutable_names(mutable_toolsets, available_names)
        requested_allowed = posture.allowed_tool_names(mutable, mutable_names)
        allowed = frozenset(requested_allowed & available_names)
        binding = posture.make_binding(
            peer, slug, context_id, allowed, mutation_enabled=mutable,
        )
        # Peer is part of the immutable binding value, not the lookup key;
        # otherwise a second peer could establish a parallel binding for the
        # same served context instead of triggering a mismatch rejection.
        binding_key = (slug, context_id)
        context_preexisted = False
        if context_from_peer:
            try:
                context_preexisted = bool(protocol.load_conversation(
                    context_id, limit=1, peer=peer, agent_slug=slug,
                ))
            except Exception:
                context_preexisted = True
            if not context_preexisted:
                try:
                    prior_tasks, _ = self.tasks.list(
                        context_id=context_id,
                        page_size=1,
                        agent_slug=self._scope_for_agent(agent)[0],
                        tenant=self._scope_for_agent(agent)[1],
                    )
                    context_preexisted = bool(prior_tasks)
                except Exception:
                    context_preexisted = True
        status, persisted_binding = posture.claim_persisted_binding(
            slug,
            context_id,
            binding,
            create_if_missing=not context_preexisted,
        )
        if status == "mismatch":
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "A2A context posture binding mismatch; start a new context.",
            ), None
        if status == "missing":
            mutable = False
            # A pre-existing transcript without an ownership binding cannot
            # safely expose history to whichever peer arrives next.
            allowed = frozenset(
                available_names
                & (posture.READONLY_TOOL_NAMES - {"a2a_history", "a2a_list"})
            )
            binding = posture.make_binding(
                peer, slug, context_id, allowed, mutation_enabled=False,
            )
        elif status in {"corrupt", "error"}:
            # Storage uncertainty never preserves mutable authority. Continue
            # only on the closed read-only surface; do not overwrite corrupt
            # evidence or pretend the binding became durable.
            mutable = False
            allowed = frozenset(
                available_names
                & (posture.READONLY_TOOL_NAMES - {"a2a_history", "a2a_list"})
            )
            binding = posture.make_binding(
                peer, slug, context_id, allowed, mutation_enabled=False,
            )
        elif persisted_binding is not None:
            binding = persisted_binding
        with self._posture_lock:
            self._posture_bindings[binding_key] = binding
        text = protocol.extract_text(params)

        # Anti-loop ping-pong protection
        turn = self._turns.track(context_id)
        if turn > protocol.max_pingpong_turns():
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)",
                           context_id, turn, protocol.max_pingpong_turns())
            rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
            self.tasks.complete(task_id, protocol.STATE_REJECTED, "")
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                f"Anti-loop protection: context {context_id} exceeded "
                f"{protocol.max_pingpong_turns()} turns. Start a new context or "
                f"increase A2A_MAX_PINGPONG_TURNS.",
                created_at=rec["created_iso"],
            ), None

        if not text:
            rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
            self.tasks.complete(task_id, protocol.STATE_REJECTED, "")
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "Empty task — nothing to do.", created_at=rec["created_iso"],
            ), None

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(
            context_id, "user", text, task_id, peer=peer, agent_slug=slug,
        )
        protocol.metrics.inbound_total += 1

        rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
        self._register_inline_push(task_id, params, agent=agent, peer=peer)

        if not agent.get("local", True):
            reply, state = self._forward_to_profile(
                agent, peer, context_id, framed, task_id=task_id,
                posture_policy={
                    "authenticated": True,
                    "served_agent_slug": slug,
                    "context_id": context_id,
                    "mutation_enabled": mutable,
                    "allowed_tool_names": sorted(allowed),
                    "binding": binding,
                },
            )
            self.tasks.complete(task_id, state, reply)
            protocol.persist_message(
                context_id, "agent", reply, task_id, peer=peer, agent_slug=slug,
            )
            security.audit("outbound", peer, task_id, reply)
            if state == protocol.STATE_COMPLETED:
                protocol.metrics.outbound_total += 1
                protocol.metrics.tasks_completed += 1
            else:
                protocol.metrics.tasks_failed += 1
            self._send_push_notification(task_id, context_id, reply, state)
            return protocol.build_task(task_id, context_id, state, reply, created_at=rec["created_iso"]), None

        if self._loop is None or self._message_handler is None:
            self.tasks.complete(task_id, protocol.STATE_FAILED, "")
            protocol.metrics.tasks_failed += 1
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED,
                "Agent gateway not ready to accept A2A tasks.",
                created_at=rec["created_iso"],
            ), None

        fut = self._add_pending(
            task_id, context_id, peer=peer, agent_slug=slug,
        )

        source = self.build_source(
            chat_id=context_id,
            chat_name=f"a2a:{peer}",
            chat_type="dm",
            user_id=peer,
            user_name=peer,
        )
        source = replace(
            source,
            a2a_mutation_requested=request.requested,
            a2a_mutation_enabled=mutable,
            a2a_credential_authenticated=credential_authenticated,
            a2a_peer_trusted=trusted,
            a2a_peer=str(peer),
            a2a_agent_slug=slug,
            a2a_context_id=context_id,
            a2a_mutable_toolsets=mutable_toolsets,
            a2a_allowed_tool_names=tuple(sorted(allowed)),
            a2a_binding=binding,
            a2a_toolset_fingerprint=binding["toolset_fingerprint"],
        )
        source = self._stamp_ingress_source(source)
        event = MessageEvent(
            text=framed,
            message_type=MessageType.TEXT,
            source=source,
            message_id=task_id,
            metadata={
                "hermes.ai/mutationAllowed": request.requested,
                "a2a_posture": {
                    "mutation_enabled": mutable,
                    "allowed_tool_names": sorted(allowed),
                    "binding": binding,
                },
            },
        )

        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception as e:
            self._pop_pending(task_id)
            msg = security.redact_outbound(f"Dispatch failed: {e}")
            self.tasks.complete(task_id, protocol.STATE_FAILED, msg)
            protocol.metrics.tasks_failed += 1
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED, msg,
                created_at=rec["created_iso"],
            ), None

        self.tasks.set_state(task_id, protocol.STATE_WORKING)
        return None, {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "agent_slug": slug,
            "future": fut,
            "created_iso": rec["created_iso"],
            "started": time.time(),
        }

    def _profile_state_db(self, profile: str) -> Optional[str]:
        home = _profile_home(profile)
        if not home:
            return None
        return os.path.join(home, "state.db")

    def _lookup_forward_session(self, profile: str, title: str) -> str:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db):
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                (title,),
            ).fetchone()
            con.close()
            return str(row[0]) if row else ""
        except Exception:
            logger.debug("A2A: could not lookup forwarded session", exc_info=True)
            return ""

    def _latest_a2a_session(self, profile: str, started_after: float) -> str:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db):
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
                (started_after - 2.0,),
            ).fetchone()
            con.close()
            return str(row[0]) if row else ""
        except Exception:
            logger.debug("A2A: could not find latest forwarded session", exc_info=True)
            return ""

    def _latest_message_id(self, profile: str, session_id: str) -> Optional[int]:
        """Return the current durable transcript boundary for a profile session."""
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return None
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            con.close()
            return int(row[0]) if row else 0
        except Exception:
            logger.debug("A2A: could not read forwarded transcript boundary", exc_info=True)
            return None

    def _latest_assistant_content(
        self, profile: str, session_id: str, after_id: int = 0
    ) -> str:
        """Return visible assistant content persisted after a transcript boundary."""
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT content FROM messages "
                "WHERE session_id = ? AND role = 'assistant' "
                "AND TRIM(COALESCE(content, '')) != '' "
                "AND COALESCE(active, 1) = 1 "
                "AND id > ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id, after_id),
            ).fetchone()
            con.close()
            return str(row[0]).strip() if row and row[0] is not None else ""
        except Exception:
            logger.debug("A2A: could not read forwarded reply", exc_info=True)
            return ""

    def _forward_tool_call_count(
        self, profile: str, session_id: str, after_id: int = 0
    ) -> Optional[int]:
        """Return how many tool calls the forwarded turn made, or None if unreadable.

        Prefers per-turn evidence (tool rows persisted after the transcript
        boundary, so a resumed session's earlier turns do not vouch for this
        one); falls back to the session-level ``tool_call_count`` counter.
        """
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return None
        con = None
        try:
            con = sqlite3.connect(db, timeout=5)
            try:
                row = con.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE session_id = ? AND id > ? "
                    "AND (role = 'tool' OR TRIM(COALESCE(tool_calls, '')) NOT IN ('', '[]', 'null'))",
                    (session_id, after_id),
                ).fetchone()
                if row is not None:
                    return int(row[0])
            except sqlite3.OperationalError:
                # Legacy/foreign schema without messages.tool_calls — fall back
                # to the session-level counter (cumulative across turns).
                logger.debug("A2A: per-turn tool-call query unavailable", exc_info=True)
            row = con.execute(
                "SELECT tool_call_count FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None or row[0] is None:
                return None
            return int(row[0])
        except Exception:
            logger.debug("A2A: could not read forwarded tool_call_count", exc_info=True)
            return None
        finally:
            if con is not None:
                con.close()

    def _title_forward_session(self, profile: str, session_id: str, title: str) -> None:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return
        try:
            con = sqlite3.connect(db, timeout=5)
            con.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            con.commit()
            con.close()
        except Exception:
            logger.debug("A2A: could not title forwarded session", exc_info=True)

    def _end_forward_session(self, profile: str, session_id: str, reason: str) -> bool:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return False
        con = None
        try:
            con = sqlite3.connect(db, timeout=5)
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND (ended_at IS NULL OR end_reason IN ('agent_close', 'cli_close'))",
                (time.time(), reason, session_id),
            )
            row = con.execute(
                "SELECT end_reason FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            con.commit()
            return bool(row and row[0] == reason)
        except Exception:
            logger.debug("A2A: could not finalize forwarded session", exc_info=True)
            return False
        finally:
            if con is not None:
                con.close()

    @staticmethod
    def _terminate_profile_process_tree(proc: subprocess.Popen) -> None:
        """Terminate and reap a timed-out Hermes child and all descendants."""
        try:
            import psutil

            parent = psutil.Process(proc.pid)
            targets = parent.children(recursive=True) + [parent]
            for target in targets:
                try:
                    target.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(targets, timeout=2)
            for target in alive:
                try:
                    target.kill()
                except psutil.NoSuchProcess:
                    pass
            if alive:
                psutil.wait_procs(alive, timeout=2)
        except Exception:
            logger.debug("A2A: process-tree termination failed", exc_info=True)
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _run_profile_command(self, cmd: list[str], timeout: int, env: dict) -> tuple[int, str, str]:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._terminate_profile_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except Exception:
                stdout = exc.output or ""
                stderr = exc.stderr or ""
            raise subprocess.TimeoutExpired(
                cmd, timeout, output=stdout, stderr=stderr
            ) from None
        except Exception:
            self._terminate_profile_process_tree(proc)
            raise
        return proc.returncode, stdout or "", stderr or ""

    @staticmethod
    def _nonzero_exit_reply(slug: str, returncode: int, stderr: str) -> str:
        """Structured failure text for a forwarded CLI that exited nonzero.

        Only the last non-empty stderr line (capped at 200 chars) is quoted —
        stdout is never returned as if it were the profile's answer.
        """
        last = ""
        for line in reversed((stderr or "").splitlines()):
            if line.strip():
                last = line.strip()
                break
        last = security.redact_outbound(last)[:200]
        detail = f": {last}" if last else ""
        return f"[profile {slug} failed rc={returncode}{detail}]"

    def _forward_to_profile(
        self, agent: dict, peer: str, context_id: str, framed_text: str, task_id: str = "",
        posture_policy: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Forward a routed A2A task to another local Hermes profile.

        First contact creates a normal ``source=a2a`` CLI session, records its
        session id, and titles it deterministically. Later turns resume by the
        concrete session id, not by a non-existent name. The public CLI boundary
        is preserved while giving A2A contexts stable multi-turn continuity.
        """
        profile = str(agent.get("profile") or agent.get("slug") or "").strip()
        slug = str(agent.get("slug") or profile or "agent")
        safe_ctx = _safe_context_slug(context_id)
        session_title = f"a2a-{slug}-{safe_ctx}"
        # Cache identity uses the raw context id; the sanitized title is only
        # presentation and may never merge distinct remote contexts.
        key = (profile or "default", slug, str(context_id))
        timeout = int(agent.get("timeout") or _reply_timeout())

        lock = self._forward_lock(profile or "default")
        with lock:
            session_id = self._profile_sessions.get(key) or self._lookup_forward_session(profile, session_title)
            message_watermark = (
                self._latest_message_id(profile, session_id) if session_id else 0
            )
            if message_watermark is None:
                return (
                    "[profile transcript boundary unavailable]",
                    protocol.STATE_FAILED,
                )
            cmd = [
                "hermes", "--profile", profile or "default",
                "chat", "-q", framed_text, "-Q", "--source", "a2a",
            ]
            model = str(agent.get("model") or "").strip()
            provider = str(agent.get("provider") or "").strip()
            if model:
                cmd.extend(["-m", model])
            if provider:
                cmd.extend(["--provider", provider])
            if session_id:
                cmd.extend(["--resume", session_id])

            env = os.environ.copy()
            # A forwarded child must not inherit dispatcher/task injection
            # capability from the parent process.
            for key in list(env):
                if key.startswith("HERMES_KANBAN_"):
                    env.pop(key, None)
            home = _profile_home(profile)
            if home:
                env["HERMES_HOME"] = home
            env["HERMES_SESSION_SOURCE"] = "a2a"
            env.pop(posture.CHILD_POLICY_ENV, None)
            if posture_policy is not None:
                # Dedicated structured policy channel; unlike the historical
                # peer hint this carries an issuer-authenticated bound surface.
                signed_policy = posture.sign_child_policy(posture_policy)
                env[posture.CHILD_POLICY_ENV] = json.dumps(
                    signed_policy, sort_keys=True, separators=(",", ":")
                )
            start = time.time()
            try:
                returncode, stdout, stderr = self._run_profile_command(cmd, timeout, env)
            except subprocess.TimeoutExpired:
                if not session_id:
                    session_id = self._latest_a2a_session(profile, start)
                if session_id:
                    self._profile_sessions[key] = session_id
                    self._title_forward_session(profile, session_id, session_title)
                    self._end_forward_session(profile, session_id, "a2a_timeout")
                return "[profile did not reply in time]", protocol.STATE_FAILED
            except Exception as e:
                if not session_id:
                    session_id = self._latest_a2a_session(profile, start)
                if session_id:
                    self._profile_sessions[key] = session_id
                    self._title_forward_session(profile, session_id, session_title)
                    self._end_forward_session(profile, session_id, "a2a_failed")
                return security.redact_outbound(f"Profile dispatch failed: {e}"), protocol.STATE_FAILED
            if returncode != 0:
                if not session_id:
                    session_id = self._latest_a2a_session(profile, start)
                if session_id:
                    self._profile_sessions[key] = session_id
                    self._title_forward_session(profile, session_id, session_title)
                    self._end_forward_session(profile, session_id, "a2a_failed")
                # Never surface stdout as an answer: a failed CLI run prints
                # its session banner (``session_id: …``) and that used to be
                # returned as if it were the reply.
                return self._nonzero_exit_reply(slug, returncode, stderr), protocol.STATE_FAILED
            if not session_id:
                session_id = self._latest_a2a_session(profile, start)
                if session_id:
                    self._profile_sessions[key] = session_id
                    self._title_forward_session(profile, session_id, session_title)
            reply = self._latest_assistant_content(
                profile, session_id, after_id=message_watermark
            )
            if not reply:
                self._end_forward_session(profile, session_id, "a2a_reply_missing")
                return "[profile produced no persisted reply]", protocol.STATE_FAILED
            if self._require_tools:
                tool_calls = self._forward_tool_call_count(
                    profile, session_id, after_id=message_watermark
                )
                if not tool_calls:
                    detail = "0 tool calls" if tool_calls == 0 else "tool_call_count unreadable"
                    reason = "HOLLOW: profile produced no tool-backed evidence"
                    logger.warning(
                        "A2A: forwarded task to profile %r (agent %r, session %s) %s — %s; "
                        "reply discarded (%d chars). Set platforms.a2a.extra.require_tools=false to allow.",
                        profile, slug, session_id, detail, reason, len(reply),
                    )
                    security.audit(
                        "hollow", peer, task_id or "",
                        f"{reason} ({detail}); profile={profile} agent={slug} session={session_id}",
                    )
                    self._end_forward_session(profile, session_id, "a2a_hollow")
                    return f"[profile {slug} HOLLOW: {detail}]", protocol.STATE_FAILED
            if not self._end_forward_session(profile, session_id, "a2a_complete"):
                return "[profile reply session could not be finalized]", protocol.STATE_FAILED
            return security.redact_outbound(reply), protocol.STATE_COMPLETED

    def _finalize_task(self, pending: dict, state: str, reply: str) -> tuple[str, str]:
        """Record the outcome of a dispatched task. Returns (state, reply) after
        redaction and input-required detection."""
        task_id = pending["task_id"]
        context_id = pending["context_id"]
        peer = pending["peer"]
        agent_slug = pending.get("agent_slug", "")
        self._pop_pending(task_id)

        reply = security.redact_outbound(reply or "")

        # The agent flags clarification requests with a leading marker; map
        # them to the A2A input-required state so the peer knows to answer.
        if state == protocol.STATE_COMPLETED:
            stripped = reply.lstrip()
            if stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
                state = protocol.STATE_INPUT_REQUIRED
                reply = stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()

        protocol.persist_message(
            context_id, "agent", reply, task_id,
            peer=peer, agent_slug=agent_slug,
        )
        security.audit("outbound", peer, task_id, reply)

        if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
            protocol.metrics.outbound_total += 1
            protocol.metrics.tasks_completed += 1
            protocol.metrics.record_latency(time.time() - pending["started"])
        else:
            protocol.metrics.tasks_failed += 1

        self.tasks.complete(task_id, state, reply)
        self._send_push_notification(task_id, context_id, reply, state)
        return state, reply

    def _await_reply(self, pending: dict, keepalive=None) -> tuple[str, str]:
        """Block until the task's future resolves (or times out).

        ``keepalive`` is an optional zero-arg callable invoked every
        _SSE_KEEPALIVE seconds while waiting (used by the SSE paths); if it
        raises, the client is gone and we stop waiting.
        """
        fut: Future = pending["future"]
        deadline = pending["started"] + _reply_timeout()
        while True:
            try:
                return fut.result(timeout=_SSE_KEEPALIVE if keepalive else max(0.0, deadline - time.time()))
            except FuturesTimeout:
                if time.time() >= deadline:
                    return (protocol.STATE_FAILED, "[agent did not reply in time]")
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        return (protocol.STATE_FAILED, "[client disconnected]")
            except Exception:
                return (protocol.STATE_FAILED, "[agent did not reply in time]")

    def _rpc_message_send(
        self, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None,
        v1_response: bool = False, *, credential_authenticated: bool = False,
    ) -> dict:
        terminal, pending = self._prepare_task(
            params, peer, agent=agent,
            credential_authenticated=credential_authenticated,
        )
        if terminal is not None:
            result = protocol.send_message_response(terminal) if v1_response else terminal
            return protocol.jsonrpc_result(req_id, result)
        state, reply = self._await_reply(pending)
        state, reply = self._finalize_task(pending, state, reply)
        task = protocol.build_task(
            pending["task_id"], pending["context_id"], state, reply,
            created_at=pending["created_iso"],
        )
        result = protocol.send_message_response(task) if v1_response else task
        return protocol.jsonrpc_result(req_id, result)

    # ── Streaming (SSE) ───────────────────────────────────────────────────

    @staticmethod
    def _sse_headers(handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        # v1.0: closing the stream signals the terminal state, so the socket
        # must actually close once we emit the done event.
        handler.close_connection = True

    @staticmethod
    def _sse_write(handler, chunk: str) -> None:
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    def _emit_terminal(self, handler, task_id: str, context_id: str, state: str, reply: str,
                       req_id: Any = None) -> None:
        """Emit the final artifact/status events and close the stream (v1.0:
        closure signals terminal state, no ``final`` field).

        ``req_id`` is threaded into JSON-RPC-wrapped SSE frames per §9.4."""
        if reply and state == protocol.STATE_COMPLETED:
            self._sse_write(handler, protocol.sse_data(
                protocol.artifact_update(task_id, context_id, reply), req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state), req_id))
        else:
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state, reply), req_id))
        self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(
        self, handler, req_id: Any, params: dict, peer: str,
        agent: Optional[dict] = None, *, credential_authenticated: bool = False,
    ) -> None:
        """Handle message/stream as an SSE response of JSON-RPC-wrapped
        StreamResponse events (A2A v1.0 §9.4)."""
        protocol.metrics.streams_started += 1
        self._sse_headers(handler)

        try:
            terminal, pending = self._prepare_task(
                params, peer, agent=agent,
                credential_authenticated=credential_authenticated,
            )
            if terminal is not None:
                self._emit_terminal(
                    handler, terminal["id"], terminal["contextId"],
                    terminal["status"]["state"],
                    protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}),
                    req_id=req_id,
                )
                return

            task_id, context_id = pending["task_id"], pending["context_id"]
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(
                protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])),
                req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))

            state, reply = self._await_reply(
                pending, keepalive=lambda: self._sse_write(handler, ": keepalive\n\n"))
            state, reply = self._finalize_task(pending, state, reply)
            self._emit_terminal(handler, task_id, context_id, state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: stream client disconnected")

    def _rpc_tasks_subscribe(self, handler, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> None:
        """Reconnect to an existing task's stream (v1.0 SubscribeToTask)."""
        task_id = str(params.get("taskId") or params.get("id") or "")
        scope = self._scope_for_agent(agent)
        rec = self.tasks.get(task_id, *scope, peer)
        if not rec:
            handler._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}"))
            return

        self._sse_headers(handler)
        try:
            fut = self.tasks.watch(task_id, *scope, peer)
            if fut is None:
                self._sse_write(handler, protocol.sse_done())
                return
            deadline = time.time() + _reply_timeout()
            while True:
                try:
                    state, reply = fut.result(timeout=_SSE_KEEPALIVE)
                    break
                except FuturesTimeout:
                    if time.time() >= deadline:
                        state, reply = rec["state"], rec.get("reply", "")
                        break
                    self._sse_write(handler, ": keepalive\n\n")
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")

    # ── Task queries ──────────────────────────────────────────────────────

    def _rpc_tasks_get(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent), peer)
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec, history_length=history_len))

    def _rpc_tasks_list(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        try:
            offset = int(params.get("pageToken") or 0)
        except (ValueError, TypeError):
            offset = 0
        try:
            page_size = int(params.get("pageSize") or 50)
        except (ValueError, TypeError):
            page_size = 50
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""),
            state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size,
            offset=max(0, offset),
            agent_slug=self._scope_for_agent(agent)[0],
            tenant=self._scope_for_agent(agent)[1],
            peer=peer,
            with_total=True,
        )
        include_artifacts = bool(params.get("includeArtifacts", False))
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, {
            "tasks": [protocol.TaskStore.to_task(r, history_length=history_len, include_artifacts=include_artifacts) for r in recs],
            "nextPageToken": str(next_offset) if next_offset else "",
            "pageSize": max(1, min(page_size, 100)),
            "totalSize": total,
        })

    def _rpc_tasks_cancel(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent), peer)
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        if rec["state"] in protocol.TERMINAL_STATES:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_CANCELABLE,
                f"task {task_id} already {rec['state']}")
        self.tasks.complete(task_id, protocol.STATE_CANCELED, "")
        self._turns.reset(rec["context_id"])
        self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent), peer) or rec
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec))

    # ── Push notifications ────────────────────────────────────────────────

    def _register_inline_push(self, task_id: str, params: dict, agent: Optional[dict] = None, peer: str = "") -> None:
        """v1.0: message/send can carry configuration.taskPushNotificationConfig."""
        cfg = (params.get("configuration") or {}).get("taskPushNotificationConfig") or {}
        if not isinstance(cfg, dict):
            return
        url = cfg.get("url") or (cfg.get("pushNotificationConfig") or {}).get("url") or ""
        if url:
            self.tasks.set_push_config(task_id, str(url), *self._scope_for_agent(agent), peer)

    def _rpc_push_config_create(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        task_id = str(params.get("taskId") or "")
        cfg = params.get("pushNotificationConfig") or params.get("config") or {}
        url = str((cfg or {}).get("url") or "")
        if not task_id or not url:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS,
                "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope_for_agent(agent), peer)
        if stored is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        return protocol.jsonrpc_result(req_id, stored)

    def _rpc_push_config_get(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        """GetTaskPushNotificationConfig — retrieve a push config by task id."""
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        cfg = self.tasks.get_push_config(task_id, config_id, *self._scope_for_agent(agent), peer)
        if cfg is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, cfg)

    def _rpc_push_config_list(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        """ListTaskPushNotificationConfigs — list push configs for a task."""
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        configs = self.tasks.list_push_configs(task_id, *self._scope_for_agent(agent), peer)
        return protocol.jsonrpc_result(req_id, {"configs": configs, "nextPageToken": ""})

    def _rpc_push_config_delete(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        """DeleteTaskPushNotificationConfig — remove a push config."""
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        deleted = self.tasks.delete_push_config(task_id, config_id, *self._scope_for_agent(agent), peer)
        if not deleted:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, {"deleted": True})

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str) -> None:
        """POST a v1.0 StreamResponse payload to the task's registered callback.

        Validates the callback URL to prevent SSRF — blocks internal/private
        addresses (169.254.x.x metadata, loopback, RFC1918 private ranges)
        unless we're in localhost-only mode (where internal access is expected).
        """
        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return

        if not security.is_safe_callback_url(callback_url):
            logger.warning("A2A: push notification for task %s blocked — unsafe callback URL: %s",
                           task_id, callback_url)
            protocol.metrics.push_failed += 1
            return

        # Push payload uses the StreamResponse format (same as streaming).
        payload = protocol.status_update(
            task_id, context_id, state, security.redact_outbound(reply or "")[:2000],
        )

        signature = security.sign_push_payload(payload)
        headers = {"Content-Type": "application/json"}
        if signature:
            headers["X-A2A-Signature"] = signature

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(callback_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    protocol.metrics.push_sent += 1
                    logger.debug("A2A: push notification sent for task %s", task_id)
                else:
                    protocol.metrics.push_failed += 1
                    logger.warning("A2A: push notification for task %s got HTTP %d", task_id, resp.status)
        except Exception as e:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s failed: %s", task_id, e)

    # ── Sending (the agent's reply path) ──────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Fulfil the pending reply Future for this context.

        ``chat_id`` is the A2A context id we set as the source chat_id. Reply
        metadata must also carry the immutable peer + served-route scope; the
        oldest outstanding task in that exact scope receives the reply.

        The gateway marks final user-visible replies with ``metadata['notify']``
        (see ``_mark_notify_metadata`` in gateway.platforms.base — this is the
        base adapter's documented reply marker, not an incidental field).
        Progress, status, and editable preview sends intentionally lack the
        marker; those must not satisfy the JSON-RPC caller.
        """
        message_id = str(int(time.time() * 1000))
        if not (metadata or {}).get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
            return SendResult(success=True, message_id=message_id)
        meta = metadata or {}
        peer = meta.get("a2a_peer")
        agent_slug = meta.get("a2a_agent_slug")
        bound_context = meta.get("a2a_context_id")
        if (
            not isinstance(peer, str)
            or not isinstance(agent_slug, str)
            or not isinstance(bound_context, str)
            or bound_context != str(chat_id)
        ):
            logger.warning("A2A: dropping final send without exact reply scope")
            return SendResult(success=False, error="missing or mismatched A2A reply scope")
        if not self._resolve_oldest_for_scope(
            peer, agent_slug, bound_context,
            protocol.STATE_COMPLETED, content or "",
        ):
            # No waiter (e.g. a late chunk or out-of-band send) — drop it.
            logger.debug("A2A: send() for scope %r had no pending waiter", (peer, agent_slug, bound_context))
        return SendResult(success=True, message_id=message_id)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Resolve the task future when processing ends without a reply send.

        The success path resolves via send(); this hook catches failures,
        cancellations, and empty runs so the HTTP thread returns promptly
        instead of waiting out the reply timeout.
        """
        task_id = str(getattr(event, "message_id", "") or "")
        if not task_id:
            return
        if outcome == ProcessingOutcome.FAILURE:
            self._resolve_task(task_id, protocol.STATE_FAILED, "[agent processing failed]")
        elif outcome == ProcessingOutcome.CANCELLED:
            self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        else:
            self._resolve_task(task_id, protocol.STATE_COMPLETED, "")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}
