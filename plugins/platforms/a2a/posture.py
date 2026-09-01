"""Machine-enforced mutation posture for A2A turns.

The A2A transport is an authenticated peer boundary, not an operator
boundary.  A peer may request mutable tools, but that request is effective
only when the transport and the served route explicitly admit it.  The only
durable side effect in this module is creation of a profile-instance identity
used to prevent authority from surviving profile replacement.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import fcntl
except Exception:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt
except Exception:  # pragma: no cover - POSIX
    msvcrt = None

MUTATION_METADATA_KEY = "hermes.ai/mutationAllowed"
CHILD_POLICY_ENV = "HERMES_A2A_POSTURE"
CHILD_POLICY_ISSUER = "hermes-a2a-adapter-v1"
PROFILE_INSTANCE_ID_FILE = ".a2a-profile-instance-id"
PROFILE_INSTANCE_ID_VERSION = "profile-instance-v1"

# This is a deliberately closed initial surface.  Adding a name is a policy
# change, not an incidental schema change.
READONLY_TOOL_NAMES = frozenset(
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

# Composite tools create a second execution authority beneath the A2A turn.
# They remain unavailable until the exact signed posture can be propagated and
# revalidated inside every nested sandbox RPC, delegated child agent, or Tool
# Search bridge dispatch.
NON_TRANSITIVE_TOOL_NAMES = frozenset(
    {"execute_code", "delegate_task", "tool_call"}
)


def _platform_value(platform: Any) -> str:
    """Normalize enum and raw-string platform identities at policy seams."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


@dataclass(frozen=True)
class MutationRequest:
    requested: bool = False
    error: str | None = None


@dataclass(frozen=True)
class PostureBinding:
    peer: str
    agent_slug: str
    context_id: str
    served_profile: str
    served_tenant: str
    profile_home_identity: str
    toolset_fingerprint: str
    mutation_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "agent_slug": self.agent_slug,
            "context_id": self.context_id,
            "served_profile": self.served_profile,
            "served_tenant": self.served_tenant,
            "profile_home_identity": self.profile_home_identity,
            "toolset_fingerprint": self.toolset_fingerprint,
            "mutation_enabled": self.mutation_enabled,
        }

    @classmethod
    def from_value(cls, value: Any) -> "PostureBinding | None":
        if not isinstance(value, dict):
            return None
        fields = (
            "peer", "agent_slug", "context_id", "served_profile",
            "served_tenant", "profile_home_identity", "toolset_fingerprint",
        )
        # The default/root served agent has an intentionally empty slug.  All
        # identity and binding material other than that URL slug is mandatory.
        if any(
            not isinstance(value.get(name), str)
            or (name not in {"agent_slug", "served_tenant"} and not value.get(name))
            for name in fields
        ):
            return None
        if "mutation_enabled" not in value or type(value.get("mutation_enabled")) is not bool:
            return None
        return cls(*(value[name] for name in fields), value["mutation_enabled"])


def _metadata_values(params: dict) -> list[Any]:
    """Collect the two interoperable metadata locations without coercion."""
    values: list[Any] = []
    message = params.get("message") if isinstance(params, dict) else None
    for container in (
        message.get("metadata") if isinstance(message, dict) else None,
        params.get("metadata") if isinstance(params, dict) else None,
    ):
        if not isinstance(container, dict):
            continue
        if MUTATION_METADATA_KEY in container:
            values.append(container[MUTATION_METADATA_KEY])
    return values


def parse_mutation_request(params: dict) -> MutationRequest:
    """Parse mutation metadata strictly, rejecting malformed/conflicting data.

    JSON ``true``/``false`` are Python ``bool`` values.  ``1``, strings and
    other truthy values are intentionally not accepted.  A missing marker is
    the safe read-only posture.
    """
    values = _metadata_values(params)
    if not values:
        return MutationRequest()
    if any(type(value) is not bool for value in values):  # noqa: E721 - exact bool is the contract
        return MutationRequest(error=f"{MUTATION_METADATA_KEY} must be a JSON boolean")
    if len(set(values)) != 1:
        return MutationRequest(error=f"conflicting {MUTATION_METADATA_KEY} metadata")
    return MutationRequest(requested=values[0])


def effective_mutation(
    *,
    requested: bool,
    credential_authenticated: bool,
    trusted_peer: bool,
    mutable_toolsets: Iterable[str] | None,
) -> bool:
    """Return whether a request may use its explicitly mutable toolsets."""
    # This primitive is also called outside the JSON parser.  Never let Python
    # truthiness turn integers or strings into an authorization decision.
    if not all(
        type(value) is bool  # noqa: E721 - exact bool is the security contract
        for value in (requested, credential_authenticated, trusted_peer)
    ):
        return False
    return requested and credential_authenticated and trusted_peer and any(
        str(item).strip() for item in (mutable_toolsets or ())
    )


def allowed_tool_names(
    mutation_enabled: bool,
    mutable_names: Iterable[str] | None = None,
) -> frozenset[str]:
    """Return the closed tool-name allowlist for this turn."""
    if not mutation_enabled:
        return READONLY_TOOL_NAMES
    mutable = {str(name) for name in (mutable_names or ())}
    return frozenset((READONLY_TOOL_NAMES | mutable) - NON_TRANSITIVE_TOOL_NAMES)


def toolset_fingerprint(tool_names: Iterable[str]) -> str:
    """Stable, immutable fingerprint of the post-assembly tool-name set."""
    canonical = json.dumps(sorted({str(name) for name in tool_names}), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_profile_instance_id(path: Path) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        uid_getter = getattr(os, "getuid", None)
        owner_mismatch = uid_getter is not None and info.st_uid != uid_getter()
        if not stat.S_ISREG(info.st_mode) or owner_mismatch:
            raise PermissionError("unsafe A2A profile instance identity owner or type")
        if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("unsafe A2A profile instance identity permissions")
        with os.fdopen(fd, "r", encoding="ascii") as handle:
            fd = -1
            raw = handle.read().strip()
    finally:
        if fd >= 0:
            os.close(fd)
    parsed = uuid.UUID(raw)
    if parsed.version != 4 or str(parsed) != raw:
        raise ValueError("invalid A2A profile instance identity")
    return raw


def _load_or_create_profile_instance_id(profile_home: Path) -> str:
    path = profile_home / PROFILE_INSTANCE_ID_FILE
    try:
        return _read_profile_instance_id(path)
    except FileNotFoundError:
        value = str(uuid.uuid4())
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            return _read_profile_instance_id(path)
        try:
            payload = f"{value}\n".encode("ascii")
            written = os.write(fd, payload)
            if written != len(payload):
                raise OSError("short A2A profile instance identity write")
            os.fsync(fd)
        finally:
            os.close(fd)
        return _read_profile_instance_id(path)


def profile_home_identity(profile_home: str) -> str:
    """Return the stable opaque identity of one concrete profile instance."""
    raw = str(profile_home or "").strip()
    if not raw:
        raise ValueError("served profile home is required")
    home = Path(raw).expanduser().resolve(strict=True)
    if not home.is_dir():
        raise NotADirectoryError(f"served profile home is not a directory: {home}")
    instance_id = _load_or_create_profile_instance_id(home)
    material = f"{home}\x00{instance_id}".encode("utf-8")
    return f"{PROFILE_INSTANCE_ID_VERSION}:{hashlib.sha256(material).hexdigest()}"


def make_binding(
    peer: str,
    agent_slug: str,
    context_id: str,
    tool_names: Iterable[str],
    *,
    served_profile: str,
    served_tenant: str,
    profile_home_identity: str,
    mutation_enabled: bool = False,
) -> dict[str, Any]:
    return PostureBinding(
        peer=str(peer),
        agent_slug=str(agent_slug),
        context_id=str(context_id),
        served_profile=str(served_profile),
        served_tenant=str(served_tenant),
        profile_home_identity=str(profile_home_identity),
        toolset_fingerprint=toolset_fingerprint(tool_names),
        mutation_enabled=bool(mutation_enabled),
    ).to_dict()


def resume_binding_status(existing: Any, current: Any) -> str:
    """Compare persisted/current bindings: ``ok``, ``mismatch`` or ``corrupt``."""
    old = PostureBinding.from_value(existing)
    new = PostureBinding.from_value(current)
    if old is None or new is None:
        return "corrupt"
    return "ok" if old == new else "mismatch"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source_binding_valid(source: Any) -> bool:
    """Require all transport-derived source fields to agree with one binding.

    Live transport provenance is validated separately against the registered
    adapter's opaque capability at gateway ingress. This helper validates the
    immutable value relationship and is intentionally unable to mint trust.
    """
    binding = PostureBinding.from_value(getattr(source, "a2a_binding", None))
    if binding is None:
        return False
    requested = getattr(source, "a2a_mutation_requested", None)
    enabled = getattr(source, "a2a_mutation_enabled", None)
    authenticated = getattr(source, "a2a_credential_authenticated", None)
    trusted = getattr(source, "a2a_peer_trusted", None)
    if not all(type(value) is bool for value in (requested, enabled, authenticated, trusted)):
        return False
    if enabled and not (requested and authenticated and trusted):
        return False
    allowed = getattr(source, "a2a_allowed_tool_names", None)
    if not isinstance(allowed, (list, tuple, set, frozenset)):
        return False
    allowed_names = frozenset(str(name) for name in allowed)
    if allowed_names & NON_TRANSITIVE_TOOL_NAMES:
        return False
    if not enabled and not allowed_names.issubset(READONLY_TOOL_NAMES):
        return False
    return (
        binding.peer == str(getattr(source, "a2a_peer", ""))
        and binding.agent_slug == str(getattr(source, "a2a_agent_slug", ""))
        and binding.context_id == str(getattr(source, "a2a_context_id", ""))
        and binding.context_id == str(getattr(source, "chat_id", ""))
        and binding.toolset_fingerprint
        == str(getattr(source, "a2a_toolset_fingerprint", ""))
        and binding.toolset_fingerprint == toolset_fingerprint(allowed_names)
        and binding.mutation_enabled is enabled
    )


def readonly_source_updates(source: Any) -> dict[str, Any] | None:
    """Build a quarantined read-only replacement for one valid source.

    Missing/corrupt durable ownership cannot safely expose conversation
    discovery or history, even when the current request's context ID matches.
    """
    current = PostureBinding.from_value(getattr(source, "a2a_binding", None))
    if current is None:
        return None
    allowed = sorted(
        set(getattr(source, "a2a_allowed_tool_names", ()) or ())
        & (READONLY_TOOL_NAMES - {"a2a_history", "a2a_list"})
    )
    binding = make_binding(
        current.peer,
        current.agent_slug,
        current.context_id,
        allowed,
        served_profile=current.served_profile,
        served_tenant=current.served_tenant,
        profile_home_identity=current.profile_home_identity,
        mutation_enabled=False,
    )
    return {
        "a2a_mutation_enabled": False,
        "a2a_allowed_tool_names": tuple(allowed),
        "a2a_toolset_fingerprint": binding["toolset_fingerprint"],
        "a2a_binding": binding,
    }


def redact_audit_summary(summary: Any, limit: int = 500) -> str:
    """Redact sensitive material *before* applying the audit length limit."""
    try:
        from .security import redact_outbound

        safe = redact_outbound(str(summary or ""))
    except Exception:
        # Never make a redaction failure an audit exfiltration path.
        safe = "[audit summary redaction failed]"
    return safe[: max(0, int(limit))]


def resolve_mutable_names(toolsets: Iterable[str] | None, available: Iterable[str]) -> frozenset[str]:
    """Resolve explicit mutable toolsets, accepting already-resolved names too."""
    available_set = {str(name) for name in (available or ())}
    resolved: set[str] = set()
    try:
        from model_tools import resolve_toolset
    except Exception:
        resolve_toolset = None
    for raw in toolsets or ():
        name = str(raw).strip()
        if not name:
            continue
        if name in available_set:
            resolved.add(name)
        if resolve_toolset is not None:
            try:
                resolved.update(set(resolve_toolset(name)) & available_set)
            except Exception:
                pass
    return frozenset(resolved - NON_TRANSITIVE_TOOL_NAMES)


def apply_to_agent(agent: Any, source: Any) -> frozenset[str] | None:
    """Bind and filter an already fully assembled agent schema.

    The function is intentionally idempotent and also narrows an accidentally
    wide cached agent.  It returns the final name set, or ``None`` for a
    non-A2A source.
    """
    if _platform_value(getattr(source, "platform", None)) != "a2a":
        return None
    binding_valid = source_binding_valid(source)
    mutable_toolsets = getattr(source, "a2a_mutable_toolsets", ()) or ()
    requested = getattr(source, "a2a_mutation_requested", False)
    credential_authenticated = getattr(source, "a2a_credential_authenticated", False)
    trusted_peer = getattr(source, "a2a_peer_trusted", False)
    if not all(type(value) is bool for value in (requested, credential_authenticated, trusted_peer)):
        requested = credential_authenticated = trusted_peer = False
    computed_enabled = effective_mutation(
        requested=requested,
        credential_authenticated=credential_authenticated,
        trusted_peer=trusted_peer,
        mutable_toolsets=mutable_toolsets,
    )
    # The adapter's binding decision is authoritative for resumed contexts;
    # never recompute a wider posture from the request marker alone.
    enabled_marker = getattr(source, "a2a_mutation_enabled", None)
    # Effective posture is an intersection: a forged/malformed source marker
    # cannot widen a request that failed credential, route, or toolset policy.
    enabled = (
        binding_valid
        and computed_enabled
        and type(enabled_marker) is bool
        and enabled_marker is True
    )
    bound_allowed = frozenset(
        str(name) for name in (getattr(source, "a2a_allowed_tool_names", ()) or ())
    )
    allowed = bound_allowed if binding_valid else frozenset()
    if getattr(agent, "tools", None) is not None:
        agent.tools = [
            td for td in agent.tools
            if td.get("function", {}).get("name") in allowed
        ]
    agent.valid_tool_names = {
        td.get("function", {}).get("name")
        for td in (getattr(agent, "tools", None) or [])
        if isinstance(td, dict) and isinstance(td.get("function"), dict)
    }
    # A cached agent must never retain a wider previous posture.
    agent._a2a_allowed_tool_names = frozenset(agent.valid_tool_names & allowed)
    agent.valid_tool_names = set(agent._a2a_allowed_tool_names)
    agent._a2a_posture = {
        "mutation_enabled": enabled,
        "allowed_tool_names": agent._a2a_allowed_tool_names,
        "binding_fingerprint": getattr(source, "a2a_toolset_fingerprint", None),
        "binding": getattr(source, "a2a_binding", None),
    }
    agent._a2a_posture_request_key = request_key(source)
    agent._a2a_toolset_fingerprint = toolset_fingerprint(agent.valid_tool_names)
    expected_fingerprint = getattr(source, "a2a_toolset_fingerprint", None)
    agent._a2a_posture_binding_matches = bool(
        binding_valid
        and expected_fingerprint
        and expected_fingerprint == agent._a2a_toolset_fingerprint
    )
    return agent._a2a_allowed_tool_names


def request_key(source: Any) -> tuple[Any, ...]:
    """Stable cache key for all posture inputs that can widen/narrow tools."""
    mutable_toolsets = getattr(source, "a2a_mutable_toolsets", ()) or ()
    binding = PostureBinding.from_value(getattr(source, "a2a_binding", None))
    return (
        getattr(source, "a2a_mutation_requested", False) if type(getattr(source, "a2a_mutation_requested", False)) is bool else False,
        getattr(source, "a2a_mutation_enabled", False) if type(getattr(source, "a2a_mutation_enabled", False)) is bool else False,
        getattr(source, "a2a_credential_authenticated", False) if type(getattr(source, "a2a_credential_authenticated", False)) is bool else False,
        getattr(source, "a2a_peer_trusted", False) if type(getattr(source, "a2a_peer_trusted", False)) is bool else False,
        tuple(sorted(str(item) for item in mutable_toolsets)),
        tuple(sorted(str(item) for item in (getattr(source, "a2a_allowed_tool_names", ()) or ()))),
        str(getattr(source, "a2a_peer", "")),
        str(getattr(source, "a2a_agent_slug", "")),
        str(getattr(source, "a2a_context_id", "")),
        binding.served_profile if binding is not None else "",
        binding.served_tenant if binding is not None else "",
        binding.profile_home_identity if binding is not None else "",
    )


def tool_allowed(agent: Any, name: str) -> bool:
    posture = getattr(agent, "_a2a_posture", None)
    if posture is None:
        return True
    normalized = str(name)
    return (
        normalized not in NON_TRANSITIVE_TOOL_NAMES
        and normalized in posture.get("allowed_tool_names", ())
    )


def _child_issuer_key_path() -> Path:
    """Fixed same-user installation key; deliberately independent of profiles."""
    from hermes_constants import get_default_hermes_root

    return Path(get_default_hermes_root()) / "a2a_child_issuer.key"


def _lock_fd(fd: int) -> None:
    """Take one blocking cross-process lock without assuming POSIX APIs."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return
    if msvcrt is not None:
        # msvcrt locks a byte range from the current position and refuses an
        # empty file. All A2A lock files therefore carry one inert byte.
        if os.fstat(fd).st_size == 0:
            os.write(fd, b" ")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("cross-process file locking is unavailable")


def _unlock_fd(fd: int) -> None:
    """Release a lock acquired by _lock_fd."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _load_or_create_child_issuer_key() -> bytes:
    path = _child_issuer_key_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, lock_flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(lock_fd, 0o600)
        _lock_fd(lock_fd)
        if not path.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags, 0o600)
            try:
                key = secrets.token_bytes(32)
                written = os.write(fd, key)
                if written != len(key):
                    raise OSError("short A2A child issuer key write")
                os.fsync(fd)
            finally:
                os.close(fd)
        info = path.lstat()
        uid_getter = getattr(os, "getuid", None)
        owner_mismatch = uid_getter is not None and info.st_uid != uid_getter()
        if not stat.S_ISREG(info.st_mode) or owner_mismatch:
            raise PermissionError("unsafe A2A child issuer key owner or type")
        if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("unsafe A2A child issuer key permissions")
        key = path.read_bytes()
        if len(key) != 32:
            raise ValueError("invalid A2A child issuer key")
        return key
    finally:
        _unlock_fd(lock_fd)
        os.close(lock_fd)


def _child_policy_payload(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if key != "signature"}


def sign_child_policy(policy: Any, *, secret: bytes | None = None) -> dict[str, Any]:
    """Sign one adapter-issued child policy with the installation key."""
    if not isinstance(policy, dict):
        raise ValueError("invalid forwarded A2A posture policy")
    payload = dict(policy)
    binding = PostureBinding.from_value(payload.get("binding"))
    if binding is not None:
        payload.setdefault("served_profile", binding.served_profile)
        payload.setdefault("served_tenant", binding.served_tenant)
        payload.setdefault("profile_home_identity", binding.profile_home_identity)
    payload["issuer"] = CHILD_POLICY_ISSUER
    payload.pop("signature", None)
    key = secret if secret is not None else _load_or_create_child_issuer_key()
    payload["signature"] = hmac.new(
        key, _canonical_json(payload), hashlib.sha256,
    ).hexdigest()
    return payload


def load_child_policy(
    value: Any = None, *, secret: bytes | None = None,
) -> dict[str, Any] | None:
    """Parse the authenticated forwarded-child policy channel strictly."""
    raw = os.environ.get(CHILD_POLICY_ENV) if value is None else value
    if not raw:
        return None
    try:
        policy = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"error": "invalid forwarded A2A posture policy"}
    if not isinstance(policy, dict):
        return {"error": "invalid forwarded A2A posture policy"}
    if policy.get("issuer") != CHILD_POLICY_ISSUER:
        return {"error": "untrusted forwarded A2A posture issuer"}
    signature = policy.get("signature")
    if not isinstance(signature, str):
        return {"error": "unsigned forwarded A2A posture policy"}
    try:
        key = secret if secret is not None else _load_or_create_child_issuer_key()
        expected_signature = hmac.new(
            key,
            _canonical_json(_child_policy_payload(policy)),
            hashlib.sha256,
        ).hexdigest()
    except Exception:
        return {"error": "forwarded A2A posture issuer unavailable"}
    if not hmac.compare_digest(signature, expected_signature):
        return {"error": "invalid forwarded A2A posture signature"}
    binding = PostureBinding.from_value(policy.get("binding"))
    allowed = policy.get("allowed_tool_names")
    if binding is None or not isinstance(allowed, list) or any(not isinstance(v, str) for v in allowed):
        return {"error": "invalid forwarded A2A posture policy"}
    if binding.toolset_fingerprint != toolset_fingerprint(allowed):
        return {"error": "forwarded A2A posture fingerprint mismatch"}
    if (
        type(policy.get("mutation_enabled")) is not bool
        or policy.get("authenticated") is not True
        or policy.get("served_agent_slug") != binding.agent_slug
        or policy.get("context_id") != binding.context_id
        or policy.get("served_profile") != binding.served_profile
        or policy.get("served_tenant") != binding.served_tenant
        or policy.get("profile_home_identity") != binding.profile_home_identity
    ):
        return {"error": "invalid forwarded A2A posture policy"}
    if binding.mutation_enabled is not policy["mutation_enabled"]:
        return {"error": "forwarded A2A posture decision mismatch"}
    if set(allowed) & NON_TRANSITIVE_TOOL_NAMES:
        return {"error": "forwarded A2A policy contains unbounded composite tools"}
    if not policy["mutation_enabled"] and not set(allowed).issubset(READONLY_TOOL_NAMES):
        return {"error": "read-only forwarded A2A policy contains mutable tools"}
    return {
        "issuer": CHILD_POLICY_ISSUER,
        "signature": signature,
        "authenticated": True,
        "served_agent_slug": policy["served_agent_slug"],
        "context_id": policy["context_id"],
        "served_profile": policy["served_profile"],
        "served_tenant": policy["served_tenant"],
        "profile_home_identity": policy["profile_home_identity"],
        "mutation_enabled": policy["mutation_enabled"],
        "allowed_tool_names": sorted(set(allowed)),
        "binding": binding.to_dict(),
    }


def child_policy_matches_tools(policy: Any, tool_names: Iterable[str]) -> bool:
    """Confirm the child's concrete post-assembly schema matches its binding."""
    if not isinstance(policy, dict):
        return False
    binding = PostureBinding.from_value(policy.get("binding"))
    return bool(
        binding is not None
        and binding.toolset_fingerprint == toolset_fingerprint(tool_names)
    )


def _binding_store_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:
        home = Path.home() / ".hermes"
    return home / "a2a_posture_bindings.json"


def _legacy_binding_valid(value: Any) -> bool:
    """Recognize pre-route-identity bindings only as fail-closed tombstones."""
    if not isinstance(value, dict) or all(
        name in value
        for name in ("served_profile", "served_tenant", "profile_home_identity")
    ):
        return False
    fields = ("peer", "agent_slug", "context_id", "toolset_fingerprint")
    return bool(
        all(
            isinstance(value.get(name), str)
            and (name == "agent_slug" or bool(value.get(name)))
            for name in fields
        )
        and type(value.get("mutation_enabled")) is bool
    )


def _read_bindings_unlocked(path: Path) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
    if not path.exists():
        return "ok", {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "corrupt", {}
    if not isinstance(data, dict):
        return "corrupt", {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or "\x00" in key or not isinstance(value, dict):
            return "corrupt", {}
        slug, sep, context = key.partition("\x1f")
        if not sep or not context or (
            PostureBinding.from_value(value) is None and not _legacy_binding_valid(value)
        ):
            return "corrupt", {}
        out[(slug, context)] = value
    return "ok", out


def _write_bindings_unlocked(
    path: Path, bindings: dict[tuple[str, str], dict[str, Any]],
) -> None:
    payload = {
        f"{slug}\x1f{context}": value
        for (slug, context), value in bindings.items()
        if PostureBinding.from_value(value) is not None or _legacy_binding_valid(value)
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _open_binding_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    _lock_fd(fd)
    return fd


def load_persisted_bindings() -> dict[tuple[str, str], dict[str, Any]]:
    """Load a validated posture snapshot under the cross-process lock."""
    path = _binding_store_path()
    fd = None
    try:
        fd = _open_binding_lock(path)
        status, bindings = _read_bindings_unlocked(path)
        return bindings if status == "ok" else {}
    except Exception:
        return {}
    finally:
        if fd is not None:
            _unlock_fd(fd)
            os.close(fd)


def claim_persisted_binding(
    slug: str, context_id: str, candidate: Any, *, create_if_missing: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    """Atomically compare-and-set one binding across adapter processes."""
    parsed = PostureBinding.from_value(candidate)
    if parsed is None or parsed.agent_slug != str(slug) or parsed.context_id != str(context_id):
        return "corrupt", None
    path = _binding_store_path()
    fd = None
    try:
        fd = _open_binding_lock(path)
        status, bindings = _read_bindings_unlocked(path)
        if status != "ok":
            return "corrupt", None
        key = (str(slug), str(context_id))
        existing = bindings.get(key)
        if existing is not None:
            if _legacy_binding_valid(existing):
                return "legacy", existing
            comparison = resume_binding_status(existing, parsed.to_dict())
            return comparison, existing
        if not create_if_missing:
            return "missing", None
        bindings[key] = parsed.to_dict()
        _write_bindings_unlocked(path, bindings)
        return "claimed", parsed.to_dict()
    except Exception:
        return "error", None
    finally:
        if fd is not None:
            _unlock_fd(fd)
            os.close(fd)


def persist_bindings(bindings: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Atomically replace a validated snapshot under the process lock."""
    path = _binding_store_path()
    fd = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = _open_binding_lock(path)
        _write_bindings_unlocked(path, bindings)
    except Exception:
        return
    finally:
        if fd is not None:
            _unlock_fd(fd)
            os.close(fd)
