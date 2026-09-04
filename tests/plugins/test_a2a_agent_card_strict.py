"""Strict-decoder proof for emitted A2A Agent Cards (t_00a4aacb).

The emitter in plugins/platforms/a2a/protocol.py builds cards by hand
(stdlib only, no a2a-sdk dependency at runtime). This suite proves the
emitted cards conform to the OFFICIAL A2A v1 protobuf schema by parsing
them with the reference implementation — the `a2a-sdk` package
(PyPI: a2a-sdk, google.protobuf-backed `a2a.types.AgentCard`) — with
unknown fields REJECTED (json_format strict mode, the decoder default).

Coverage:
  * Representative authenticated and unauthenticated cards strict-parse
    against the official proto and round-trip their key fields.
  * Unknown-field drift is rejected (no permissive local schema here).
  * Drift control: stateTransitionHistory — previously emitted, now
    non-canonical — IS rejected by the strict decoder, proving the gate
    catches exactly the regression class this change removes.
  * Wire-level parity: the card actually served over HTTP by the live
    adapter strict-parses too.
  * Behavior-preservation guard: pre-fix cards FAIL strict parse, so the
    tests below cannot pass vacuously against the old emitter.

Skips (not fails) when a2a-sdk is not installed in the test venv.
"""

from __future__ import annotations

import copy
import json

import pytest

json_format = pytest.importorskip(
    "google.protobuf.json_format", reason="a2a-sdk (protobuf) not installed"
)
a2a_types = pytest.importorskip("a2a.types", reason="a2a-sdk not installed")
json_format.Parse  # attribute exists ⇒ protobuf runtime present

from plugins.platforms.a2a import protocol

# Known pre-1.0 convenience members our emitter deliberately keeps for
# legacy peers (documented in protocol.build_agent_card). The official v1
# proto carries url only inside supportedInterfaces[], so these are
# stripped before the strict decode — this is the emitter's documented
# contract, not schema leniency: after stripping, the decoder runs in
# strict mode (ignore_unknown_fields=False, the json_format default).
_LEGACY_ALIASES = ("url", "security")


def _strict_parse(card: dict) -> "a2a_types.AgentCard":
    """Strict-decode a card against the official A2A v1 AgentCard proto."""
    canon = {k: v for k, v in card.items() if k not in _LEGACY_ALIASES}
    return json_format.Parse(json.dumps(canon), a2a_types.AgentCard())


class TestStrictDecoderAuthenticated:
    def test_authenticated_card_strict_parses(self):
        card = protocol.build_agent_card(
            name="hermes-test",
            url="http://localhost:9900/",
            description="test",
            skills=[{"id": "general", "name": "general",
                     "description": "general agent", "tags": ["general"]}],
            streaming=True,
            push_notifications=True,
            auth_required=True,
            tenant="dev-team",
        )
        msg = _strict_parse(card)
        assert msg.name == "hermes-test"
        assert (
            msg.security_schemes["bearer"].http_auth_security_scheme.scheme
            == "Bearer"
        )
        assert len(msg.security_requirements) == 1
        assert list(msg.security_requirements[0].schemes["bearer"].list) == []
        assert msg.supported_interfaces[0].protocol_binding == "JSONRPC"
        assert msg.supported_interfaces[0].tenant == "dev-team"
        assert msg.capabilities.streaming is True
        assert msg.capabilities.push_notifications is True

    def test_authenticated_card_carries_no_legacy_security(self):
        card = protocol.build_agent_card(
            name="x", url="u", description="d", auth_required=True,
        )
        assert "security" not in card
        assert card["securityRequirements"] == [
            {"schemes": {"bearer": {"list": []}}}
        ]


class TestStrictDecoderUnauthenticated:
    def test_unauthenticated_card_strict_parses(self):
        card = protocol.build_agent_card(
            name="local",
            url="http://127.0.0.1:9900/",
            description="localhost card",
            auth_required=False,
        )
        msg = _strict_parse(card)
        assert msg.name == "local"
        # Unauthenticated localhost card: no security requirements,
        # no security schemes.
        assert len(msg.security_requirements) == 0
        assert len(msg.security_schemes) == 0
        assert "securityRequirements" not in card
        assert "securitySchemes" not in card


class TestUnknownFieldDriftRejected:
    def test_arbitrary_unknown_field_rejected(self):
        card = protocol.build_agent_card(
            name="x", url="u", description="d", auth_required=True,
        )
        canon = {k: v for k, v in card.items() if k not in _LEGACY_ALIASES}
        drifted = {**canon, "bogusField": 1}
        with pytest.raises(Exception):
            json_format.Parse(json.dumps(drifted), a2a_types.AgentCard())

    def test_legacy_security_field_rejected_by_strict_decoder(self):
        """Pre-1.0 singular `security` is drift the strict decoder refuses —
        which is exactly why the emitter no longer produces it."""
        card = protocol.build_agent_card(
            name="x", url="u", description="d", auth_required=True,
        )
        canon = {k: v for k, v in card.items() if k != "url"}
        drifted = {**canon, "security": [{"bearer": []}]}
        with pytest.raises(Exception):
            json_format.Parse(json.dumps(drifted), a2a_types.AgentCard())

    def test_state_transition_history_is_rejected_drift(self):
        """The field this change stops emitting must itself be drift."""
        card = protocol.build_agent_card(
            name="x", url="u", description="d",
        )
        canon = {k: v for k, v in card.items() if k not in _LEGACY_ALIASES}
        assert "stateTransitionHistory" not in card["capabilities"]
        drifted = copy.deepcopy(canon)
        drifted["capabilities"]["stateTransitionHistory"] = False
        with pytest.raises(Exception):
            json_format.Parse(json.dumps(drifted), a2a_types.AgentCard())


class TestBehaviorPreservationGuard:
    """If the emitter regressed to the pre-fix shape, strict parse must fail —
    so the suite cannot pass vacuously."""

    def test_pre_fix_card_fails_strict_parse(self):
        pre_fix_card = {
            "name": "x",
            "description": "d",
            "version": "1.0.0",
            "provider": {"organization": "Hermes Agent", "url": "u"},
            "supportedInterfaces": [{
                "url": "u", "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }],
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,  # pre-fix non-canonical member
                "extendedAgentCard": False,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [],
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "security": [{"bearer": []}],
        }
        # Strip only the documented legacy url alias; the rest is the old
        # shape verbatim and must NOT strict-parse.
        canon = {k: v for k, v in pre_fix_card.items() if k != "url"}
        with pytest.raises(Exception):
            json_format.Parse(json.dumps(canon), a2a_types.AgentCard())


class TestServedCardWireParity:
    def test_live_served_card_strict_parses(self, monkeypatch):
        """The bytes actually served over HTTP strict-parse, not just the
        in-process dict."""
        import asyncio
        import json as _json
        import socket
        import urllib.request

        from gateway.config import PlatformConfig
        from plugins.platforms.a2a.adapter import A2AAdapter

        monkeypatch.setenv("A2A_BEARER_TOKEN", "strict-fixture-token")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        monkeypatch.setenv("A2A_PORT", str(port))

        adapter = A2AAdapter(PlatformConfig(enabled=True))

        async def run():
            assert await adapter.connect() is True
            try:
                url = f"http://127.0.0.1:{port}/.well-known/agent-card.json"
                req = urllib.request.Request(
                    url, headers={"Authorization": "Bearer strict-fixture-token"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return _json.loads(resp.read().decode())
            finally:
                await adapter.disconnect()

        card = asyncio.run(run())
        msg = _strict_parse(card)
        assert msg.name
        assert msg.supported_interfaces[0].url
        # Served with a token configured ⇒ authenticated card on the wire.
        assert len(msg.security_requirements) == 1
        assert (
            msg.security_schemes["bearer"].http_auth_security_scheme.scheme
            == "Bearer"
        )
        assert "security" not in card
        assert "stateTransitionHistory" not in card["capabilities"]
