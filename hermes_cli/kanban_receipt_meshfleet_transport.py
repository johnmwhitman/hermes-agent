"""Native stdio transport for the Hermes Kanban receipt outbox.

The transport starts the *explicitly supplied* installed MeshFleet entrypoint
with the *explicitly supplied* Node executable and storage paths. It never
discovers a global checkout or falls back to a mock.
Every stdio session observes and validates the running server's identity, tool
catalog, and storage schema before its requested tool is called.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_cli.kanban_receipt import (
    WORK_RECEIPT_SOURCE,
    compute_payload_sha256,
    validate_envelope,
)


_MCP_RECORD = "record_work_receipt"
_MCP_GET = "get_work_receipt"
_MCP_V3 = "verify_ledger_v3"
_MCP_HEALTH = "get_health"
_REQUIRED_TOOLS = frozenset({_MCP_RECORD, _MCP_GET, _MCP_V3, _MCP_HEALTH})
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class LiveMcpNotConfigured(RuntimeError):
    """The supplied installed consumer cannot be trusted or started."""


@dataclass(frozen=True)
class McpRuntimeConfig:
    """All paths and identity expectations required for a native session.

    There are intentionally no machine-specific defaults. In particular,
    ``expected_storage_schema_version`` is an explicit observation target,
    rather than the obsolete assumption that every acceptable consumer is v5.
    """

    node_bin: Path
    entrypoint: Path
    meshfleet_db_file: Path
    meshfleet_data_file: Path
    agent_mesh_data_file: Path
    event_log_file: Path
    artifact_dir: Path
    expected_source_ref: str
    expected_package_name: str
    expected_package_version: str
    expected_storage_schema_version: int

    def __post_init__(self) -> None:
        for field_name in (
            "node_bin",
            "entrypoint",
            "meshfleet_db_file",
            "meshfleet_data_file",
            "agent_mesh_data_file",
            "event_log_file",
            "artifact_dir",
        ):
            value = Path(getattr(self, field_name)).expanduser()
            if not value.is_absolute():
                raise LiveMcpNotConfigured(f"{field_name} must be an absolute path")
            object.__setattr__(self, field_name, value)
        if not self.node_bin.is_file():
            raise LiveMcpNotConfigured(f"Node executable is not a file: {self.node_bin}")
        if not os.access(self.node_bin, os.X_OK):
            raise LiveMcpNotConfigured(f"Node executable is not executable: {self.node_bin}")
        if not self.entrypoint.is_file():
            raise LiveMcpNotConfigured(
                f"installed MeshFleet entrypoint is not a file: {self.entrypoint}"
            )
        if not _HEX_40.fullmatch(self.expected_source_ref):
            raise LiveMcpNotConfigured("expected_source_ref must be a 40-char lowercase git SHA")
        if not self.expected_package_name.strip() or not self.expected_package_version.strip():
            raise LiveMcpNotConfigured("expected package name and version must be non-empty")
        if (
            not isinstance(self.expected_storage_schema_version, int)
            or isinstance(self.expected_storage_schema_version, bool)
            or self.expected_storage_schema_version < 1
        ):
            raise LiveMcpNotConfigured(
                "expected_storage_schema_version must be a positive integer"
            )


@dataclass(frozen=True)
class ToolReply:
    payload: dict[str, Any]
    is_error: bool
    raw_text: str


@dataclass(frozen=True)
class SessionExchange:
    server_name: str
    server_version: str
    catalog: dict[str, dict[str, Any]]
    health: ToolReply
    result: ToolReply | None


@dataclass(frozen=True)
class LiveGateStatus:
    """Observed identity of an exact stdio consumer process."""

    ready: bool
    node_version: str
    health_status: str
    server_name: str
    server_version: str
    source_ref: str
    package_name: str
    package_version: str
    storage_schema_version: int
    tool_names: tuple[str, ...]
    manifest_path: str


SessionCall = Callable[[McpRuntimeConfig, str | None, dict[str, Any]], SessionExchange]
NodeVersionProbe = Callable[[Path], str]
StorageProbe = Callable[[Path], int]


class MeshFleetMcpTransport:
    """Synchronous receipt transport backed by the Python MCP SDK stdio client.

    The optional callables are test seams only. The installed CLI constructs
    this class without them, which necessarily uses ``stdio_client`` and
    ``ClientSession``. No success path is synthesized when the SDK, process,
    catalog, identity, or database marker is unavailable.
    """

    def __init__(
        self,
        config: McpRuntimeConfig,
        *,
        session_call: SessionCall | None = None,
        node_version_probe: NodeVersionProbe | None = None,
        storage_probe: StorageProbe | None = None,
    ) -> None:
        self.config = config
        self._session_call = session_call or _real_stdio_session_call
        self._node_version_probe = node_version_probe or _read_node_version
        self._storage_probe = storage_probe or _read_storage_schema_version
        self.record_call_count = 0
        self.readback_call_count = 0
        self.audit_call_count = 0
        self.last_observation: LiveGateStatus | None = None
        self.last_accepted_receipt: dict[str, Any] | None = None
        self.last_readback_receipt: dict[str, Any] | None = None
        self.last_audit_handle: str | None = None

    def probe(self) -> LiveGateStatus:
        """Start the configured consumer and validate observed runtime state."""
        self._exchange(None, {})
        assert self.last_observation is not None
        return self.last_observation

    def record_work_receipt(self, envelope: dict[str, Any]) -> Any:
        (
            conflict,
            duplicate,
            recorded,
            _transport_error,
            validation_failed,
            RecordReceiptResult,
            _VerifyLedgerResult,
        ) = _outcomes()
        try:
            reply = self._exchange(_MCP_RECORD, dict(envelope))
        except Exception as exc:
            raise _transport_unavailable_cls()(str(exc)) from exc
        self.record_call_count += 1
        assert reply is not None
        payload = reply.payload
        if reply.is_error:
            error = payload.get("error")
            if error == "work_receipt_conflict":
                return RecordReceiptResult(
                    outcome=conflict,
                    error=str(payload.get("detail") or error),
                )
            if isinstance(error, str) and error.startswith(
                "record_work_receipt: invalid input"
            ):
                return RecordReceiptResult(outcome=validation_failed, error=error)
            raise _transport_unavailable_cls()(
                f"record_work_receipt failed: {error or payload!r}"
            )

        inserted = payload.get("inserted")
        replayed = payload.get("replayed")
        if (
            set(payload) != {"ok", "inserted", "replayed", "recorded_at", "receipt"}
            or payload.get("ok") is not True
            or not isinstance(inserted, bool)
            or not isinstance(replayed, bool)
            or inserted == replayed
        ):
            raise _transport_unavailable_cls()(
                f"record_work_receipt returned invalid booleans/shape: {payload!r}"
            )
        recorded_at = _positive_int(payload.get("recorded_at"), "recorded_at")
        wire_receipt = payload.get("receipt")
        if not isinstance(wire_receipt, dict) or any(
            key in wire_receipt for key in ("source", "recorded_at")
        ):
            raise _transport_unavailable_cls()(
                "record_work_receipt must return caller-owned receipt fields"
            )
        # The record RPC returns its validated input plus an outer timestamp.
        # Only get_work_receipt returns the persisted source/timestamp fields.
        receipt = _validate_server_receipt({
            **wire_receipt,
            "source": WORK_RECEIPT_SOURCE,
            "recorded_at": recorded_at,
        })
        accepted_id = _derive_accepted_id(receipt)
        if accepted_id is None:
            raise _transport_unavailable_cls()(
                "record_work_receipt returned no validated receipt identity"
            )
        self.last_accepted_receipt = dict(receipt)
        return RecordReceiptResult(
            outcome=duplicate if replayed else recorded,
            accepted_id=accepted_id,
            accepted_receipt=receipt,
            recorded_at=recorded_at,
        )

    def get_work_receipt(self, task_id: str, run_id: int) -> dict[str, Any] | None:
        try:
            reply = self._exchange(
                _MCP_GET,
                {
                    "source": WORK_RECEIPT_SOURCE,
                    "task_id": task_id,
                    "run_id": run_id,
                },
            )
        except Exception as exc:
            raise _transport_unavailable_cls()(str(exc)) from exc
        self.readback_call_count += 1
        assert reply is not None
        if reply.is_error:
            error = reply.payload.get("error")
            if isinstance(error, str) and error.startswith(
                "get_work_receipt: no work receipt"
            ):
                return None
            raise _transport_unavailable_cls()(
                f"get_work_receipt failed: {error or reply.payload!r}"
            )
        if set(reply.payload) != {"ok", "receipt"} or reply.payload.get("ok") is not True:
            raise _transport_unavailable_cls()(
                f"get_work_receipt returned invalid response: {reply.payload!r}"
            )
        receipt = _validate_server_receipt(reply.payload.get("receipt"))
        self.last_readback_receipt = dict(receipt)
        return receipt

    def verify_ledger_v3(self) -> Any:
        *_, VerifyLedgerResult = _outcomes()
        try:
            reply = self._exchange(_MCP_V3, {})
        except Exception as exc:
            raise _transport_unavailable_cls()(str(exc)) from exc
        self.audit_call_count += 1
        assert reply is not None
        if reply.is_error:
            raise _transport_unavailable_cls()(
                f"verify_ledger_v3 failed: {reply.payload.get('error') or reply.payload!r}"
            )
        payload = reply.payload
        if (
            set(payload)
            != {"schema", "evidence_scope", "report", "finding_local_bands"}
            or payload.get("schema") != "meshfleet.verify/v3"
        ):
            raise _transport_unavailable_cls()(
                f"unexpected verifier schema: {payload.get('schema')!r}"
            )
        report = payload.get("report")
        if not isinstance(report, dict) or not isinstance(report.get("ok"), bool):
            raise _transport_unavailable_cls()("verify_ledger_v3 response lacks report.ok boolean")
        artifact_path, artifact_sha = _write_audit_artifact(
            self.config.artifact_dir, reply.raw_text
        )
        raw_findings = report.get("findings") or []
        if not isinstance(raw_findings, list):
            raw_findings = [raw_findings]
        errors = [
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            if isinstance(item, (dict, list))
            else str(item)
            for item in raw_findings
        ]
        audit_handle = f"sha256:{artifact_sha};path:{artifact_path}"
        self.last_audit_handle = audit_handle
        return VerifyLedgerResult(
            ok=report["ok"],
            audit_handle=audit_handle,
            errors=errors,
        )

    def _exchange(
        self, tool_name: str | None, args: dict[str, Any]
    ) -> ToolReply | None:
        if tool_name is not None and tool_name not in _REQUIRED_TOOLS:
            raise LiveMcpNotConfigured(f"unexpected MCP tool call: {tool_name!r}")
        node_version = self._node_version_probe(self.config.node_bin)
        if not re.fullmatch(r"v24(?:\.\d+){1,2}", node_version):
            raise LiveMcpNotConfigured(
                f"configured Node runtime is not Node 24: {node_version!r}"
            )
        exchange = self._session_call(self.config, tool_name, args)
        storage_version = self._storage_probe(self.config.meshfleet_db_file)
        self.last_observation = _validate_observation(
            self.config, node_version, storage_version, exchange
        )
        return exchange.result


def _outcomes() -> Any:
    from hermes_cli.kanban_receipt_reconciler import (
        DELIVERY_OUTCOME_CONFLICT,
        DELIVERY_OUTCOME_DUPLICATE,
        DELIVERY_OUTCOME_RECORDED,
        DELIVERY_OUTCOME_TRANSPORT_ERROR,
        DELIVERY_OUTCOME_VALIDATION_FAILED,
        RecordReceiptResult,
        VerifyLedgerResult,
    )

    return (
        DELIVERY_OUTCOME_CONFLICT,
        DELIVERY_OUTCOME_DUPLICATE,
        DELIVERY_OUTCOME_RECORDED,
        DELIVERY_OUTCOME_TRANSPORT_ERROR,
        DELIVERY_OUTCOME_VALIDATION_FAILED,
        RecordReceiptResult,
        VerifyLedgerResult,
    )


def _transport_unavailable_cls() -> Any:
    from hermes_cli.kanban_receipt_reconciler import TransportUnavailable

    return TransportUnavailable


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _transport_unavailable_cls()(f"{name} must be a positive integer")
    return value


def _validate_server_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _transport_unavailable_cls()("consumer response lacks receipt object")
    expected_keys = {
        "source",
        "schema",
        "task_id",
        "run_id",
        "assignee",
        "terminal_outcome",
        "result_contract",
        "quality_gate",
        "completed_at",
        "evidence",
        "payload_sha256",
        "recorded_at",
    }
    if set(value) != expected_keys:
        raise _transport_unavailable_cls()(
            f"consumer receipt fields mismatch: {sorted(value)}"
        )
    receipt = dict(value)
    if receipt.get("source") != WORK_RECEIPT_SOURCE:
        raise _transport_unavailable_cls()("consumer receipt source mismatch")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(entry, dict)
        or set(entry) not in ({"kind", "handle"}, {"kind", "handle", "digest"})
        for entry in evidence
    ):
        raise _transport_unavailable_cls()("consumer receipt evidence fields mismatch")
    _positive_int(receipt.get("recorded_at"), "receipt.recorded_at")
    reasons = validate_envelope(receipt)
    if reasons:
        raise _transport_unavailable_cls()(
            "consumer returned invalid receipt: " + "; ".join(reasons)
        )
    recomputed = compute_payload_sha256(receipt)
    if receipt.get("payload_sha256") != recomputed:
        raise _transport_unavailable_cls()(
            "consumer receipt payload_sha256 does not match canonical fields"
        )
    return receipt


def _derive_accepted_id(receipt: dict[str, Any]) -> str | None:
    """Derive internal provenance only after a real receipt is validated."""
    try:
        receipt = _validate_server_receipt(receipt)
    except Exception:
        return None
    digest = receipt["payload_sha256"]
    if not _HEX_64.fullmatch(digest):
        return None
    return (
        f"{receipt['source']}\x00{receipt['task_id']}\x00"
        f"{receipt['run_id']}\x00{digest}"
    )


def _validate_observation(
    config: McpRuntimeConfig,
    node_version: str,
    storage_version: int,
    exchange: SessionExchange,
) -> LiveGateStatus:
    if exchange.server_name != "agent-mesh":
        raise LiveMcpNotConfigured(
            f"unexpected MCP server name: {exchange.server_name!r}"
        )
    if exchange.server_version != config.expected_package_version:
        raise LiveMcpNotConfigured(
            "MCP initialize version does not match expected package version"
        )
    _validate_catalog(exchange.catalog)
    if exchange.health.is_error:
        raise LiveMcpNotConfigured(
            f"get_health returned MCP error: {exchange.health.payload!r}"
        )
    health = exchange.health.payload
    health_status = health.get("status")
    if health_status not in {"ok", "degraded"}:
        raise LiveMcpNotConfigured(
            f"MeshFleet get_health status is unusable: {health_status!r}"
        )
    if (
        not isinstance(health.get("work_receipt_count"), int)
        or isinstance(health.get("work_receipt_count"), bool)
        or health["work_receipt_count"] < 0
    ):
        raise LiveMcpNotConfigured("get_health lacks work_receipt_count integer")
    identity = health.get("build_identity")
    if not isinstance(identity, dict):
        raise LiveMcpNotConfigured("get_health lacks build_identity")
    expected_identity = {
        "status": "ok",
        "schema": "meshfleet.build/v1",
        "package_name": config.expected_package_name,
        "package_version": config.expected_package_version,
        "source_commit": config.expected_source_ref,
        "entrypoints_match_runtime": True,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected or (
            key == "entrypoints_match_runtime"
            and not isinstance(identity.get(key), bool)
        ):
            raise LiveMcpNotConfigured(
                f"build_identity {key} mismatch: expected {expected!r}, "
                f"observed {identity.get(key)!r}"
            )
    manifest = identity.get("manifest_path")
    if not isinstance(manifest, str):
        raise LiveMcpNotConfigured("build_identity lacks manifest_path")
    expected_manifest = (config.entrypoint.parent / "meshfleet-build-manifest.json").resolve()
    if Path(manifest).resolve() != expected_manifest:
        raise LiveMcpNotConfigured(
            "build_identity manifest_path is not beside the supplied entrypoint"
        )
    if not expected_manifest.is_file():
        raise LiveMcpNotConfigured("supplied installed entrypoint has no build manifest")
    try:
        manifest_payload = json.loads(expected_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveMcpNotConfigured("installed build manifest is unreadable") from exc
    if not isinstance(manifest_payload, dict):
        raise LiveMcpNotConfigured("installed build manifest is not a JSON object")
    package = manifest_payload.get("package")
    manifest_entrypoints = manifest_payload.get("entrypoints")
    if (
        manifest_payload.get("schema") != "meshfleet.build/v1"
        or manifest_payload.get("source_commit") != config.expected_source_ref
        or not isinstance(package, dict)
        or package.get("name") != config.expected_package_name
        or package.get("version") != config.expected_package_version
        or not isinstance(manifest_entrypoints, dict)
    ):
        raise LiveMcpNotConfigured("installed build manifest identity mismatch")
    entrypoints = identity.get("entrypoints")
    entrypoint_sha = (
        str(entrypoints.get(config.entrypoint.name) or "")
        if isinstance(entrypoints, dict)
        else ""
    )
    if not _HEX_64.fullmatch(entrypoint_sha):
        raise LiveMcpNotConfigured(
            "build_identity manifest lacks the supplied runtime entrypoint"
        )
    if hashlib.sha256(config.entrypoint.read_bytes()).hexdigest() != entrypoint_sha:
        raise LiveMcpNotConfigured(
            "build_identity entrypoint digest does not match supplied entrypoint bytes"
        )
    if manifest_entrypoints.get(config.entrypoint.name) != entrypoint_sha:
        raise LiveMcpNotConfigured(
            "installed manifest and get_health disagree on runtime entrypoint digest"
        )
    if storage_version != config.expected_storage_schema_version:
        raise LiveMcpNotConfigured(
            "storage schema mismatch: expected "
            f"{config.expected_storage_schema_version}, observed {storage_version}"
        )
    return LiveGateStatus(
        ready=True,
        node_version=node_version,
        health_status=health_status,
        server_name=exchange.server_name,
        server_version=exchange.server_version,
        source_ref=identity["source_commit"],
        package_name=identity["package_name"],
        package_version=identity["package_version"],
        storage_schema_version=storage_version,
        tool_names=tuple(sorted(exchange.catalog)),
        manifest_path=str(expected_manifest),
    )


def _validate_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    missing = _REQUIRED_TOOLS - set(catalog)
    if missing:
        raise LiveMcpNotConfigured(f"MCP catalog missing required tools: {sorted(missing)}")
    record = catalog[_MCP_RECORD]
    expected_record_required = {
        "schema",
        "task_id",
        "run_id",
        "assignee",
        "terminal_outcome",
        "result_contract",
        "quality_gate",
        "completed_at",
        "evidence",
        "payload_sha256",
    }
    if (
        record.get("type") != "object"
        or set(record.get("required") or []) != expected_record_required
        or record.get("additionalProperties") is not False
        or set(record.get("properties") or {}) != expected_record_required
    ):
        raise LiveMcpNotConfigured("record_work_receipt catalog schema is not the flat envelope")
    props = record.get("properties") or {}
    evidence = props.get("evidence") or {}
    evidence_items = evidence.get("items") or {}
    evidence_props = evidence_items.get("properties") or {}
    if (
        (props.get("schema") or {}).get("type") != "string"
        or (props.get("schema") or {}).get("enum") != ["hermes.kanban-result/v1"]
        or (props.get("task_id") or {}).get("type") != "string"
        or (props.get("task_id") or {}).get("pattern") != "^t_[A-Za-z0-9]+$"
        or (props.get("run_id") or {}).get("type") != "integer"
        or (props.get("run_id") or {}).get("minimum") != 1
        or (props.get("assignee") or {}).get("type") != "string"
        or (props.get("assignee") or {}).get("minLength") != 1
        or (props.get("terminal_outcome") or {}).get("type") != "string"
        or (props.get("terminal_outcome") or {}).get("enum")
        != ["completed", "failed", "refused", "blocked"]
        or (props.get("result_contract") or {}).get("type") != "string"
        or (props.get("result_contract") or {}).get("enum")
        != ["ok", "refused", "blocked", "artifact_missing", "invalid", "absent"]
        or (props.get("quality_gate") or {}).get("type") != "string"
        or (props.get("quality_gate") or {}).get("enum") != ["passed", "failed"]
        or (props.get("completed_at") or {}).get("type") != "integer"
        or (props.get("completed_at") or {}).get("minimum") != 1
        or evidence.get("type") != "array"
        or evidence_items.get("type") != "object"
        or set(evidence_items.get("required") or []) != {"kind", "handle"}
        or evidence_items.get("additionalProperties") is not False
        or set(evidence_props) != {"kind", "handle", "digest"}
        or (evidence_props.get("kind") or {}).get("type") != "string"
        or (evidence_props.get("kind") or {}).get("enum")
        != ["git_commit", "attachment", "artifact", "command_run", "external"]
        or (evidence_props.get("handle") or {}).get("type") != "string"
        or (evidence_props.get("handle") or {}).get("minLength") != 1
        or (evidence_props.get("digest") or {}).get("type") != "string"
        or (evidence_props.get("digest") or {}).get("pattern") != "^[0-9a-f]{64}$"
        or (props.get("payload_sha256") or {}).get("type") != "string"
        or (props.get("payload_sha256") or {}).get("pattern") != "^[0-9a-f]{64}$"
    ):
        raise LiveMcpNotConfigured("record_work_receipt catalog field contract mismatch")
    get_schema = catalog[_MCP_GET]
    get_props = get_schema.get("properties") or {}
    if (
        get_schema.get("type") != "object"
        or set(get_schema.get("required") or []) != {"source", "task_id", "run_id"}
        or get_schema.get("additionalProperties") is not False
        or set(get_props) != {"source", "task_id", "run_id"}
        or (get_props.get("source") or {}).get("type") != "string"
        or (get_props.get("source") or {}).get("enum") != [WORK_RECEIPT_SOURCE]
        or (get_props.get("task_id") or {}).get("type") != "string"
        or (get_props.get("task_id") or {}).get("pattern") != "^t_[A-Za-z0-9]+$"
        or (get_props.get("run_id") or {}).get("type") != "integer"
        or (get_props.get("run_id") or {}).get("minimum") != 1
    ):
        raise LiveMcpNotConfigured("get_work_receipt catalog schema mismatch")
    verify_schema = catalog[_MCP_V3]
    if verify_schema.get("type") != "object" or verify_schema.get("properties") != {}:
        raise LiveMcpNotConfigured("verify_ledger_v3 catalog schema mismatch")
    health_schema = catalog[_MCP_HEALTH]
    health_props = health_schema.get("properties")
    # Corrected consumers support an optional verbosity selector. Our empty
    # argument call keeps the full identity response and all runtime checks.
    supported_health_props = health_props == {}
    if isinstance(health_props, dict) and set(health_props) == {"verbosity"}:
        verbosity = health_props["verbosity"]
        supported_health_props = (
            isinstance(verbosity, dict)
            and verbosity.get("type") == "string"
            and verbosity.get("enum") == ["full", "summary"]
        )
    if (
        health_schema.get("type") != "object"
        or not supported_health_props
        or ("required" in health_schema and health_schema["required"] != [])
    ):
        raise LiveMcpNotConfigured("get_health catalog schema mismatch")


def _read_node_version(node_bin: Path) -> str:
    try:
        proc = subprocess.run(
            [str(node_bin), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LiveMcpNotConfigured(f"could not execute supplied Node runtime: {exc}") from exc
    if proc.returncode != 0:
        raise LiveMcpNotConfigured(
            f"supplied Node runtime --version failed with exit {proc.returncode}"
        )
    return proc.stdout.strip()


def _read_storage_schema_version(db_file: Path) -> int:
    if not db_file.is_file():
        raise LiveMcpNotConfigured(f"MeshFleet DB was not created: {db_file}")
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'storage_schema_version'"
            ).fetchone()
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='work_receipts'"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise LiveMcpNotConfigured(f"could not observe MeshFleet storage schema: {exc}") from exc
    if row is None or table is None:
        raise LiveMcpNotConfigured("MeshFleet storage marker or work_receipts table is missing")
    try:
        value = int(row[0])
    except (TypeError, ValueError) as exc:
        raise LiveMcpNotConfigured("MeshFleet storage schema marker is invalid") from exc
    return value


def _write_audit_artifact(artifact_dir: Path, raw_text: str) -> tuple[Path, str]:
    artifact_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    stat = artifact_dir.lstat()
    if artifact_dir.is_symlink() or not artifact_dir.is_dir() or stat.st_uid != os.getuid():
        raise _transport_unavailable_cls()("audit artifact directory is not an owned directory")
    raw = raw_text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    destination = artifact_dir / f"meshfleet-verify-v3-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != raw:
            raise _transport_unavailable_cls()("audit artifact digest collision on disk")
        return destination.resolve(), digest
    fd, tmp_name = tempfile.mkstemp(prefix=".meshfleet-audit-", dir=artifact_dir)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return destination.resolve(), digest


def _tool_reply(result: Any) -> ToolReply:
    is_error = getattr(result, "is_error", None)
    if not isinstance(is_error, bool):
        raise LiveMcpNotConfigured("MCP SDK result lacks is_error boolean")
    content = getattr(result, "content", None)
    if not isinstance(content, list) or len(content) != 1:
        raise LiveMcpNotConfigured("MCP result must contain exactly one JSON text item")
    item = content[0]
    if getattr(item, "type", None) != "text" or not isinstance(
        getattr(item, "text", None), str
    ):
        raise LiveMcpNotConfigured("MCP result content is not JSON text")
    raw_text = item.text
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LiveMcpNotConfigured("MCP result text is not JSON") from exc
    if not isinstance(payload, dict):
        raise LiveMcpNotConfigured("MCP result JSON is not an object")
    return ToolReply(payload=payload, is_error=is_error, raw_text=raw_text)


def _real_stdio_session_call(
    config: McpRuntimeConfig,
    tool_name: str | None,
    args: dict[str, Any],
) -> SessionExchange:
    """Run one bounded MCP stdio session using the installed Python SDK."""
    try:
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise LiveMcpNotConfigured(
            "native receipt reconciliation requires the supported Hermes mcp extra"
        ) from exc

    async def run() -> SessionExchange:
        stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        params = StdioServerParameters(
            command=str(config.node_bin),
            args=[str(config.entrypoint)],
            env={
                "MESHFLEET_DB_FILE": str(config.meshfleet_db_file),
                "MESHFLEET_DATA_FILE": str(config.meshfleet_data_file),
                "AGENT_MESH_DATA_FILE": str(config.agent_mesh_data_file),
                "MESHFLEET_EVENT_LOG_FILE": str(config.event_log_file),
                "AGENT_MESH_CHILD": "1",
                "MESHFLEET_RATIFY_SWEEP_MS": "0",
            },
        )
        try:
            async with stdio_client(params, errlog=stderr) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=30.0) as session:
                    initialized = await session.initialize()
                    listed = await session.list_tools()
                    catalog: dict[str, dict[str, Any]] = {}
                    for tool in listed.tools:
                        dumped = tool.model_dump(by_alias=True)
                        schema = dumped.get("inputSchema")
                        if not isinstance(schema, dict):
                            raise LiveMcpNotConfigured(
                                f"tool {tool.name!r} lacks inputSchema object"
                            )
                        catalog[tool.name] = schema
                    health = _tool_reply(await session.call_tool(_MCP_HEALTH, {}))
                    observation = SessionExchange(
                        server_name=initialized.server_info.name,
                        server_version=initialized.server_info.version,
                        catalog=catalog,
                        health=health,
                        result=None,
                    )
                    # Refusing a reply after recording is too late: validate
                    # the same running session before any requested mutation.
                    _validate_observation(
                        config,
                        _read_node_version(config.node_bin),
                        _read_storage_schema_version(config.meshfleet_db_file),
                        observation,
                    )
                    result = None
                    if tool_name is not None:
                        result = _tool_reply(await session.call_tool(tool_name, args))
                    return SessionExchange(
                        server_name=initialized.server_info.name,
                        server_version=initialized.server_info.version,
                        catalog=catalog,
                        health=health,
                        result=result,
                    )
        except LiveMcpNotConfigured:
            raise
        except Exception as exc:
            stderr.seek(0)
            detail = stderr.read(4000).strip()
            suffix = f"; server stderr: {detail[-1000:]}" if detail else ""
            def error_details(error: BaseException) -> str:
                nested = getattr(error, "exceptions", ())
                return "; ".join(error_details(item) for item in nested) if nested else str(error)
            raise LiveMcpNotConfigured(
                f"MeshFleet stdio session failed: {error_details(exc)}{suffix}"
            ) from exc
        finally:
            stderr.close()

    return anyio.run(run)


def probe_live_gate(config: McpRuntimeConfig) -> LiveGateStatus:
    """Observe one exact configured consumer; no caller booleans are accepted."""
    return MeshFleetMcpTransport(config).probe()


def build_reconciler_for_live(
    *,
    conn: sqlite3.Connection,
    config: McpRuntimeConfig,
    clock: Callable[[], int],
    lease_seconds: int = 60,
) -> tuple[Any, LiveGateStatus]:
    """Build a real reconciler only after the configured consumer is observed."""
    from hermes_cli.kanban_receipt_reconciler import Reconciler

    transport = MeshFleetMcpTransport(config)
    gate = transport.probe()
    return (
        Reconciler(conn, transport, clock=clock, lease_seconds=lease_seconds),
        gate,
    )


__all__ = [
    "LiveGateStatus",
    "LiveMcpNotConfigured",
    "McpRuntimeConfig",
    "MeshFleetMcpTransport",
    "SessionExchange",
    "ToolReply",
    "build_reconciler_for_live",
    "probe_live_gate",
]
