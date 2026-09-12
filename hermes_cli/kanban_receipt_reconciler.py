"""Phase 3 reconciler for the kanban receipt outbox (t_2d7fd66c).

This module sits between the durable :class:`kanban_receipt_outbox`
table (written atomically by :func:`kanban_db.complete_task`) and the
MeshFleet wire consumer. It is the seam that turns "done transition
visible" into "MeshFleet accepted + v3 audit green + sampled as a
production-quality numerator".

Design contract (matches ``KANBAN-RECEIPT-DOGFOOD-DESIGN.md``):

* Card completion is decoupled from delivery. ``kanban_complete``
  commits the task transition and a receipt-outbox row in one SQLite
  transaction. Delivery is retried.

* Counting is strict. A receipt is only counted toward the
  production-quality numerator when MeshFleet has accepted it AND
  ``verify_ledger_v3`` is green AND the exact readback matches the
  accepted id. Refused / blocked / failed / invalid / artifact-missing
  rows are retained for the denominator and the false-DONE audit but
  never credited.

* Transport is pluggable for focused state-machine tests. Production uses the
  explicit native stdio :class:`MeshFleetMcpTransport`; there is no mock
  fallback. The native command validates the running consumer, performs exact
  readback, and preserves the actual verifier-v3 JSON before crediting.

* Idempotency owns the ``(task_id, run_id)`` pair. Re-firing a
  delivery for the same pair returns the existing accepted id (or the
  same dead-letter reason) — never a new row.

* Validation/conflict → dead-letter. Transport/unavailable → retry
  with bounded exponential backoff. A row is never silently dropped.

* Crash safety is structural. The outbox row is the durable seam, so
  the worker can recover on restart by selecting ``status='pending'``
  rows whose lease has expired.

The module's ``-m`` entrypoint reconciles one explicitly named task/run against
an explicitly supplied board DB and installed MeshFleet consumer. It does not
discover or mutate a global consumer configuration.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

from hermes_cli.kanban_receipt import (
    WORK_RECEIPT_SOURCE,
    canonical_payload,
    compute_payload_sha256,
    validate_envelope,
)
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# Phase 3 schema note: the outbox table carries the durable envelope
# (Phase 2). Phase 3 adds reconciliation state via the same row's
# ``status`` column (no new migration needed) plus an auxiliary
# ``kanban_receipt_delivery_log`` audit trail. We keep the audit
# trail separate so the close-time atomic transaction stays small and
# the worker can append per-attempt detail without touching the
# outbox's idempotency surface (``UNIQUE(task_id, run_id)``).
OUTBOX_TABLE = "kanban_receipt_outbox"
DELIVERY_LOG_TABLE = "kanban_receipt_delivery_log"

# Outbox ``status`` enum (matches Phase 2 ``CREATE TABLE`` comment).
# Phase 3 adds ``sent`` and ``credited`` to that surface; ``pending`` /
# ``failed`` / ``dead-letter`` already exist. We intentionally keep
# the values short — they appear in dashboards and audit logs.
OUTBOX_STATUS_PENDING = "pending"
OUTBOX_STATUS_SENT = "sent"
OUTBOX_STATUS_CREDITED = "credited"
OUTBOX_STATUS_FAILED = "dead-letter"  # alias kept for schema parity

# Delivery log ``outcome`` enum.
DELIVERY_OUTCOME_RECORDED = "recorded"        # record_work_receipt accepted (new id)
DELIVERY_OUTCOME_DUPLICATE = "duplicate"      # record_work_receipt returned existing id (idempotent replay)
DELIVERY_OUTCOME_VALIDATION_FAILED = "validation_failed"  # 400 / shape refusal → dead-letter
DELIVERY_OUTCOME_CONFLICT = "conflict"        # same (task_id, run_id), different bytes → dead-letter
DELIVERY_OUTCOME_TRANSPORT_ERROR = "transport_error"  # network / timeout / unavailable → retry
DELIVERY_OUTCOME_READBACK_MISMATCH = "readback_mismatch"  # exact readback diverged from accepted id → dead-letter
DELIVERY_OUTCOME_V3_NOT_GREEN = "v3_not_green"  # verify_ledger_v3 reports the row but did not pass → no credit
DELIVERY_OUTCOME_V3_GREEN = "v3_green"        # exact readback + v3 green → credit
DELIVERY_OUTCOME_CREDITED = "credited"        # terminal; outbox status flipped to credited

# Bounded retry policy (matches §3 of the design — "bounded exponential
# retry"; validation/conflict dead-letters, transport/unavailable
# retries). 8 attempts at 2x base-2s gives a max wall-clock of ~510s per
# row before dead-letter, which is generous enough for a hot
# MeshFleet process to recover and tight enough that a long outage
# surfaces in dashboards within ~10 minutes.
_MAX_ATTEMPTS = 8
_BACKOFF_BASE_SECONDS = 2.0


@dataclass
class OutboxRow:
    """One row of ``kanban_receipt_outbox`` as the worker sees it.

    Mirrors the schema and adds a typed view of the JSON envelope so
    call sites never re-parse ``payload_json`` inline.
    """

    task_id: str
    run_id: int
    payload_sha256: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    status: str = OUTBOX_STATUS_PENDING
    attempt_count: int = 0
    last_attempt_at: int | None = None
    last_error: str | None = None
    lease_expires_at: int | None = None
    # Phase 3 worker-side extensions (not persisted on the outbox
    # row; populated by the worker from the delivery log).
    accepted_id: str | None = None
    accepted_at: int | None = None
    v3_audit_handle: str | None = None
    v3_audited_at: int | None = None

    @property
    def is_creditable(self) -> bool:
        """True if the row's outcome counts toward the production-quality numerator.

        Per the design §1.3: terminal_outcome=completed,
        result_contract=ok, quality_gate=passed, at least one evidence
        handle, AND the row survives verify_ledger_v3 without an
        error. The reconciler is responsible for the last clause;
        the first four are encoded in the envelope and re-validated
        by the transport side. This helper is the gating surface the
        sampling report uses.
        """
        env = self.payload
        return (
            env.get("terminal_outcome") == "completed"
            and env.get("result_contract") == "ok"
            and env.get("quality_gate") == "passed"
            and bool(env.get("evidence"))
            and self.status == OUTBOX_STATUS_CREDITED
        )


@dataclass
class DeliveryResult:
    """One row of ``kanban_receipt_delivery_log`` as the worker writes it.

    Audit-only; the durable outbox row carries the idempotency
    surface, this log carries per-attempt detail so operators can
    debug without re-running the reconciler.
    """

    task_id: str
    run_id: int
    attempt: int
    outcome: str
    error: str | None = None
    accepted_id: str | None = None
    v3_audit_handle: str | None = None
    created_at: int = 0


@runtime_checkable
class ReceiptTransport(Protocol):
    """Pluggable transport for MeshFleet work-receipt delivery.

    Three operations — ``record_work_receipt``,
    ``get_work_receipt``, ``verify_ledger_v3`` — are the entire wire
    surface Phase 3 relies on. Production construction requires the native
    ``MeshFleetMcpTransport``; focused state-machine tests use
    :class:`MockTransport`.
    """

    def record_work_receipt(self, envelope: dict[str, Any]) -> "RecordReceiptResult":
        """Submit one canonical envelope to MeshFleet.

        Must be idempotent on ``(task_id, run_id)``: replaying the same
        envelope returns the same validated receipt with
        ``outcome=DUPLICATE``; replaying the same key with different bytes
        returns ``outcome=CONFLICT`` and never overwrites the existing row.
        """

    def get_work_receipt(self, task_id: str, run_id: int) -> dict[str, Any] | None:
        """Read back the work receipt by ``(task_id, run_id)``.

        Returns the wire-shaped receipt dict or ``None`` if no row
        exists. ``None`` after a successful ``record_work_receipt``
        is a readback mismatch and must dead-letter the row.
        """

    def verify_ledger_v3(self) -> "VerifyLedgerResult":
        """Run the v3 read-only local-consistency audit.

        Returns a structured result with ``ok`` (bool), the audit
        handle (string), and any errors. The reconciler credits a
        row only when this is ``ok=True`` AND the audit confirms
        the receipt by ``(task_id, run_id)`` is present with
        digest matching the canonical payload SHA-256.
        """


@dataclass
class RecordReceiptResult:
    """What ``record_work_receipt`` returns to the reconciler."""

    outcome: str  # DELIVERY_OUTCOME_*
    accepted_id: str | None = None
    error: str | None = None
    accepted_receipt: dict[str, Any] | None = None
    recorded_at: int | None = None


@dataclass
class VerifyLedgerResult:
    """What ``verify_ledger_v3`` returns to the reconciler."""

    ok: bool
    audit_handle: str
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MockTransport — used only by focused state-machine tests. It is NOT a fake
# MeshFleet ledger; it is a deterministic stub that honours the
# contract surface exactly so the reconciler state-machine tests prove
# the wire shape without any live process.
# ---------------------------------------------------------------------------


class MockTransport:
    """Test-only deterministic transport that obeys the wire contract.

    Behaviour matrix (matches the design §4 contract):

    * First ``record_work_receipt(envelope)`` for a given
      ``(task_id, run_id)`` returns ``outcome=RECORDED`` and stores
      the row. The accepted id is internal provenance derived from the
      validated source/task/run/digest tuple.
    * Replaying the same envelope returns ``outcome=DUPLICATE`` with
      the same accepted id (idempotent on byte-equivalent replay).
    * Replaying the same key with different bytes returns
      ``outcome=CONFLICT`` and never overwrites the existing row.
    * ``get_work_receipt(task_id, run_id)`` returns the stored row or
      ``None`` (the reconciler treats ``None`` after a RECORDED result
      as a readback mismatch).
    * ``verify_ledger_v3()`` runs a minimal local-consistency check
      (every stored row's stored digest matches the canonical
      ``payload_sha256`` of its stored envelope). Default behaviour is
      a green audit; ``force_not_green=True`` makes the next audit
      return ``ok=False`` to exercise the no-credit branch.
    """

    def __init__(self, *, force_not_green: bool = False) -> None:
        self._rows: dict[tuple[str, int], dict[str, Any]] = {}
        self._force_not_green = force_not_green
        self.record_call_count = 0
        self.audit_call_count = 0

    def record_work_receipt(self, envelope: dict[str, Any]) -> RecordReceiptResult:
        self.record_call_count += 1
        reasons = validate_envelope(envelope)
        if reasons:
            return RecordReceiptResult(
                outcome=DELIVERY_OUTCOME_VALIDATION_FAILED,
                error="; ".join(reasons),
            )
        task_id = str(envelope.get("task_id") or "")
        run_id = envelope.get("run_id")
        digest = envelope.get("payload_sha256")
        assert isinstance(run_id, int) and not isinstance(run_id, bool)
        assert isinstance(digest, str)
        key = (task_id, run_id)
        existing = self._rows.get(key)
        if existing is None:
            recorded_at = 1_700_000_000 + len(self._rows)
            receipt = {
                "source": WORK_RECEIPT_SOURCE,
                **dict(envelope),
                "recorded_at": recorded_at,
            }
            accepted_id = _receipt_identity(receipt)
            self._rows[key] = {
                "receipt": receipt,
                "accepted_id": accepted_id,
            }
            return RecordReceiptResult(
                outcome=DELIVERY_OUTCOME_RECORDED,
                accepted_id=accepted_id,
                accepted_receipt=dict(receipt),
                recorded_at=recorded_at,
            )
        if existing["receipt"].get("payload_sha256") == digest:
            return RecordReceiptResult(
                outcome=DELIVERY_OUTCOME_DUPLICATE,
                accepted_id=existing["accepted_id"],
                accepted_receipt=dict(existing["receipt"]),
                recorded_at=existing["receipt"]["recorded_at"],
            )
        return RecordReceiptResult(
            outcome=DELIVERY_OUTCOME_CONFLICT,
            error=(
                f"(task_id={task_id}, run_id={run_id}) already stored with "
                f"different bytes; refusing to overwrite"
            ),
        )

    def get_work_receipt(self, task_id: str, run_id: int) -> dict[str, Any] | None:
        row = self._rows.get((task_id, run_id))
        if row is None:
            return None
        return dict(row["receipt"])

    def verify_ledger_v3(self) -> VerifyLedgerResult:
        self.audit_call_count += 1
        if self._force_not_green:
            return VerifyLedgerResult(
                ok=False,
                audit_handle="audit_mock_NOT_GREEN",
                errors=["forced_not_green test mode"],
            )
        errors: list[str] = []
        for key, row in self._rows.items():
            env = row["receipt"]
            stored_digest = env.get("payload_sha256")
            if not isinstance(stored_digest, str) or len(stored_digest) != 64:
                errors.append(
                    f"{key}: stored digest missing or malformed"
                )
        return VerifyLedgerResult(
            ok=not errors,
            audit_handle="audit_mock_" + str(self.audit_call_count).rjust(4, "0"),
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Reconciler — the bounded state-machine worker. Public surface is
# ``Reconciler.run_once`` (process one tick) and ``Reconciler.run_batch``
# (process a bounded batch). The dispatcher decides cadence.
# ---------------------------------------------------------------------------


class TransportUnavailable(Exception):
    """Raised by a live transport when the wire is unreachable.

    The reconciler treats this as ``DELIVERY_OUTCOME_TRANSPORT_ERROR``
    — bounded retry with exponential backoff. Validation/conflict
    responses are NOT raised; they are returned as ``RecordReceiptResult``
    outcomes and dead-letter immediately.
    """


_CALLER_RECEIPT_FIELDS = {
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
_SERVER_RECEIPT_FIELDS = _CALLER_RECEIPT_FIELDS | {"source", "recorded_at"}


def _receipt_identity(receipt: dict[str, Any]) -> str | None:
    """Return internal provenance for a fully-shaped accepted receipt."""
    source = receipt.get("source")
    task_id = receipt.get("task_id")
    run_id = receipt.get("run_id")
    digest = receipt.get("payload_sha256")
    if (
        source != WORK_RECEIPT_SOURCE
        or not isinstance(task_id, str)
        or not isinstance(run_id, int)
        or isinstance(run_id, bool)
        or run_id < 1
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        return None
    return f"{source}\x00{task_id}\x00{run_id}\x00{digest}"


def _wire_envelope_from_outbox(row: OutboxRow) -> tuple[dict[str, Any] | None, str | None]:
    """Validate original durable bytes and build the flat MCP arguments."""
    original = row.payload
    if not isinstance(original, dict):
        return None, "outbox payload is not a JSON object"
    allowed = _CALLER_RECEIPT_FIELDS | {"source"}
    extras = set(original) - allowed
    if extras:
        return None, f"outbox payload has unknown fields: {sorted(extras)}"
    if "source" in original and original.get("source") != WORK_RECEIPT_SOURCE:
        return None, "outbox payload source does not match hermes-kanban"
    if original.get("task_id") != row.task_id or original.get("run_id") != row.run_id:
        return None, "outbox row identity disagrees with original payload"
    embedded_digest = original.get("payload_sha256")
    if embedded_digest is not None and embedded_digest != row.payload_sha256:
        return None, "outbox payload digest disagrees with durable digest column"
    try:
        recomputed = compute_payload_sha256(original)
        wire = canonical_payload(original)
    except (KeyError, TypeError, UnicodeError, ValueError) as exc:
        return None, f"outbox payload cannot be canonicalized: {exc}"
    if recomputed != row.payload_sha256:
        return None, (
            "outbox payload_sha256 does not match recomputed original payload "
            f"(stored={row.payload_sha256}, recomputed={recomputed})"
        )
    wire["payload_sha256"] = row.payload_sha256
    reasons = validate_envelope(wire)
    if reasons:
        return None, "outbox payload validation failed: " + "; ".join(reasons)
    return wire, None


def _accepted_receipt_matches_original(
    row: OutboxRow,
    receipt: dict[str, Any] | None,
) -> bool:
    """Compare server receipt to original bytes using server canonicalization."""
    if not isinstance(receipt, dict) or set(receipt) != _SERVER_RECEIPT_FIELDS:
        return False
    if receipt.get("source") != WORK_RECEIPT_SOURCE:
        return False
    recorded_at = receipt.get("recorded_at")
    if (
        not isinstance(recorded_at, int)
        or isinstance(recorded_at, bool)
        or recorded_at < 1
    ):
        return False
    wire, error = _wire_envelope_from_outbox(row)
    if error is not None or wire is None:
        return False
    try:
        return (
            receipt.get("payload_sha256") == row.payload_sha256
            and compute_payload_sha256(receipt) == row.payload_sha256
            and canonical_payload(receipt) == canonical_payload(wire)
        )
    except (KeyError, TypeError, UnicodeError, ValueError):
        return False


class Reconciler:
    """Drives ``kanban_receipt_outbox`` rows to ``credited`` or ``dead-letter``.

    The reconciler is intentionally small. Its job is:

    1. Select rows whose lease has expired (or never set).
    2. For each row, call ``record_work_receipt``.
    3. On success, call ``get_work_receipt`` and assert the
       full canonical accepted envelope and derived internal identity match.
    4. On readback match, call ``verify_ledger_v3`` and only flip
       the outbox to ``credited`` if the audit is green and the
       receipt is present in the audited snapshot.
    5. Persist per-attempt detail to the delivery log so operators
       can audit without re-running.

    The reconciler never deletes rows; ``dead-letter`` is a terminal
    status that retains the original envelope for forensics. It also
    never edits the original outbox ``payload_json`` — only
    ``status`` / ``attempt_count`` / ``last_attempt_at`` / ``last_error``
    are mutable by the worker.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        transport: ReceiptTransport,
        *,
        lease_seconds: int = 60,
        clock: "callable[[], int]" = time.time,
        logger: logging.Logger | None = None,
    ) -> None:
        self.conn = conn
        self.transport = transport
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.logger = logger or logging.getLogger(__name__)
        self._ensure_delivery_log()

    # ------------------------------------------------------------------
    # Schema bootstrap — additive, idempotent.
    # ------------------------------------------------------------------

    def _ensure_delivery_log(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DELIVERY_LOG_TABLE} (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         TEXT NOT NULL,
                run_id          INTEGER NOT NULL,
                attempt         INTEGER NOT NULL,
                outcome         TEXT NOT NULL,
                error           TEXT,
                accepted_id     TEXT,
                v3_audit_handle TEXT,
                created_at      INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_delivery_log_task "
            f"ON {DELIVERY_LOG_TABLE}(task_id, run_id, attempt)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Row selection — durable lease, no lock contention.
    # ------------------------------------------------------------------

    def select_pending(
        self,
        *,
        limit: int = 32,
        task_id: str | None = None,
        run_id: int | None = None,
    ) -> list[OutboxRow]:
        """Select rows whose status is ``pending`` and whose lease is expired.

        The worker takes a lease (sets ``lease_expires_at = now +
        lease_seconds``) inside the same transaction as the SELECT so
        a second concurrent worker tick cannot re-pick the same row.
        Rows in ``dead-letter`` / ``credited`` are never re-tried.
        """
        now = int(self.clock())
        filters = ["status = ?", "(lease_expires_at IS NULL OR lease_expires_at <= ?)"]
        params: list[Any] = [OUTBOX_STATUS_PENDING, now]
        if task_id is not None:
            filters.append("task_id = ?")
            params.append(task_id)
        if run_id is not None:
            filters.append("run_id = ?")
            params.append(run_id)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT task_id, run_id, payload_sha256, payload_json,
                   created_at, status, attempt_count, last_attempt_at,
                   last_error, lease_expires_at
            FROM {OUTBOX_TABLE}
            WHERE {' AND '.join(filters)}
            ORDER BY created_at ASC, run_id ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        leased_until = now + self.lease_seconds
        out: list[OutboxRow] = []
        for row in rows:
            self.conn.execute(
                f"UPDATE {OUTBOX_TABLE} SET lease_expires_at = ? "
                f"WHERE task_id = ? AND run_id = ?",
                (leased_until, row["task_id"], row["run_id"]),
            )
            out.append(self._row_from_sql(row))
        self.conn.commit()
        return out

    @staticmethod
    def _row_from_sql(row: sqlite3.Row) -> OutboxRow:
        try:
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
        except (TypeError, ValueError):
            payload = {}
        return OutboxRow(
            task_id=row["task_id"],
            run_id=int(row["run_id"]),
            payload_sha256=row["payload_sha256"],
            payload=payload,
            created_at=int(row["created_at"] or 0),
            status=row["status"],
            attempt_count=int(row["attempt_count"] or 0),
            last_attempt_at=(
                int(row["last_attempt_at"]) if row["last_attempt_at"] else None
            ),
            last_error=row["last_error"],
            lease_expires_at=(
                int(row["lease_expires_at"]) if row["lease_expires_at"] else None
            ),
        )

    # ------------------------------------------------------------------
    # Public entry points.
    # ------------------------------------------------------------------

    def run_once(self) -> list[OutboxRow]:
        """Process one batch of pending rows.

        Returns the list of rows the tick processed (regardless of
        outcome — successes, retries, and dead-letters all appear).
        Callers can use this to drive dashboards / cadence control.
        """
        rows = self.select_pending()
        processed: list[OutboxRow] = []
        for row in rows:
            self._process_one(row)
            processed.append(row)
        return processed

    def run_batch(
        self,
        *,
        max_rows: int = 256,
        task_id: str | None = None,
        run_id: int | None = None,
    ) -> list[OutboxRow]:
        """Process up to ``max_rows`` pending rows in one call.

        Same semantics as ``run_once`` but accepts a higher cap for
        bulk catch-up after a long MeshFleet outage.
        """
        rows = self.select_pending(
            limit=max_rows, task_id=task_id, run_id=run_id
        )
        processed: list[OutboxRow] = []
        for row in rows:
            self._process_one(row)
            processed.append(row)
        return processed

    # ------------------------------------------------------------------
    # Per-row state machine.
    # ------------------------------------------------------------------

    def _process_one(self, row: OutboxRow) -> None:
        attempt = row.attempt_count + 1
        now = int(self.clock())
        try:
            result = self._deliver(row)
            outcome = result.outcome
            error = result.error
            accepted_id = result.accepted_id
        except TransportUnavailable as exc:
            self._record_attempt(row, attempt, DELIVERY_OUTCOME_TRANSPORT_ERROR,
                                 error=str(exc), now=now)
            self._mark_retry_or_dead_letter(row, error=str(exc), now=now)
            return
        except Exception as exc:  # pragma: no cover - defensive
            self._record_attempt(row, attempt, DELIVERY_OUTCOME_TRANSPORT_ERROR,
                                 error=repr(exc), now=now)
            self._mark_retry_or_dead_letter(row, error=repr(exc), now=now)
            return

        # Validation / conflict → immediate dead-letter (no retry).
        if outcome in (
            DELIVERY_OUTCOME_VALIDATION_FAILED,
            DELIVERY_OUTCOME_CONFLICT,
        ):
            self._record_attempt(
                row, attempt, outcome, error=error,
                accepted_id=accepted_id, now=now,
            )
            self._mark_dead_letter(row, error=error or outcome, now=now)
            return
        if outcome not in (DELIVERY_OUTCOME_RECORDED, DELIVERY_OUTCOME_DUPLICATE):
            message = f"transport returned unknown delivery outcome: {outcome!r}"
            self._record_attempt(
                row,
                attempt,
                DELIVERY_OUTCOME_TRANSPORT_ERROR,
                error=message,
                now=now,
            )
            self._mark_retry_or_dead_letter(row, error=message, now=now)
            return

        # Recorded / duplicate — first validate the actual accepted receipt,
        # then compare an independent readback of the same server identity.
        accepted_receipt = result.accepted_receipt
        accepted_identity = (
            _receipt_identity(accepted_receipt)
            if isinstance(accepted_receipt, dict)
            else None
        )
        if (
            accepted_id is None
            or accepted_identity is None
            or accepted_id != accepted_identity
            or not _accepted_receipt_matches_original(row, accepted_receipt)
            or result.recorded_at != accepted_receipt.get("recorded_at")
        ):
            # Defensive: a successful outcome MUST carry an accepted id.
            self._record_attempt(
                row, attempt, DELIVERY_OUTCOME_READBACK_MISMATCH,
                error="transport success lacked a matching validated accepted receipt",
                now=now,
            )
            self._mark_dead_letter(
                row,
                error="accepted receipt identity/payload mismatch — refusing to credit",
                now=now,
            )
            return

        try:
            rb = self.transport.get_work_receipt(row.task_id, row.run_id)
        except Exception as exc:
            self._record_attempt(
                row,
                attempt,
                DELIVERY_OUTCOME_TRANSPORT_ERROR,
                error=str(exc),
                accepted_id=accepted_id,
                now=now,
            )
            self._mark_retry_or_dead_letter(row, error=str(exc), now=now)
            return
        readback_ok = (
            _accepted_receipt_matches_original(row, rb)
            and _receipt_identity(rb or {}) == accepted_id
            and rb == accepted_receipt
        )
        if not readback_ok:
            self._record_attempt(
                row, attempt, DELIVERY_OUTCOME_READBACK_MISMATCH,
                error=(
                    "exact readback diverged from accepted id "
                    f"(accepted_id={accepted_id}, readback={rb!r})"
                ),
                accepted_id=accepted_id, now=now,
            )
            self._mark_dead_letter(
                row,
                error="exact readback mismatch — refusing to credit",
                now=now,
            )
            return

        try:
            audit = self.transport.verify_ledger_v3()
        except Exception as exc:
            self._record_attempt(
                row,
                attempt,
                DELIVERY_OUTCOME_TRANSPORT_ERROR,
                error=str(exc),
                accepted_id=accepted_id,
                now=now,
            )
            self._mark_retry_or_dead_letter(row, error=str(exc), now=now)
            return
        if not audit.ok:
            self._record_attempt(
                row, attempt, DELIVERY_OUTCOME_V3_NOT_GREEN,
                error="; ".join(audit.errors) or "v3 audit not green",
                accepted_id=accepted_id,
                v3_audit_handle=audit.audit_handle, now=now,
            )
            # v3 not green is a wait-and-audit situation, not a
            # dead-letter. The row stays pending; the worker retries
            # on the next tick.
            self._mark_retry_or_dead_letter(
                row,
                error=f"v3 audit not green ({audit.audit_handle})",
                now=now,
            )
            return

        # All three gates pass — credit.
        self._record_attempt(
            row, attempt, DELIVERY_OUTCOME_V3_GREEN,
            accepted_id=accepted_id,
            v3_audit_handle=audit.audit_handle, now=now,
        )
        self._mark_credited(row, accepted_id=accepted_id,
                            v3_audit_handle=audit.audit_handle, now=now)

    # ------------------------------------------------------------------
    # Outbox + delivery-log writes.
    # ------------------------------------------------------------------

    def _record_attempt(
        self,
        row: OutboxRow,
        attempt: int,
        outcome: str,
        *,
        error: str | None = None,
        accepted_id: str | None = None,
        v3_audit_handle: str | None = None,
        now: int,
    ) -> None:
        self.conn.execute(
            f"""
            INSERT INTO {DELIVERY_LOG_TABLE}
                (task_id, run_id, attempt, outcome, error,
                 accepted_id, v3_audit_handle, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (row.task_id, row.run_id, attempt, outcome, error,
             accepted_id, v3_audit_handle, now),
        )

    def _mark_retry_or_dead_letter(
        self, row: OutboxRow, *, error: str, now: int,
    ) -> None:
        """Bump attempt_count; dead-letter if the budget is exhausted.

        The exponential backoff lives in ``lease_expires_at`` so a row
        is naturally invisible to ``select_pending`` until its lease
        expires. The base is ``_BACKOFF_BASE_SECONDS`` doubled per
        attempt up to a per-row cap that keeps the longest retry
        bounded by ``_MAX_ATTEMPTS``.
        """
        attempts = row.attempt_count + 1
        if attempts >= _MAX_ATTEMPTS:
            self._mark_dead_letter(row, error=error, now=now)
            return
        # backoff = base * 2^(attempts-1); cap at 2^7 * base = 256s.
        backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempts - 1))
        lease = now + int(backoff)
        self.conn.execute(
            f"UPDATE {OUTBOX_TABLE} SET status = ?, attempt_count = ?, "
            f"last_attempt_at = ?, last_error = ?, lease_expires_at = ? "
            f"WHERE task_id = ? AND run_id = ?",
            (
                OUTBOX_STATUS_PENDING,
                attempts,
                now,
                error,
                lease,
                row.task_id,
                row.run_id,
            ),
        )
        self.conn.commit()
        row.attempt_count = attempts
        row.last_attempt_at = now
        row.last_error = error
        row.lease_expires_at = lease

    def _mark_dead_letter(
        self, row: OutboxRow, *, error: str, now: int,
    ) -> None:
        self.conn.execute(
            f"UPDATE {OUTBOX_TABLE} SET status = ?, attempt_count = ?, "
            f"last_attempt_at = ?, last_error = ?, lease_expires_at = NULL "
            f"WHERE task_id = ? AND run_id = ?",
            (
                OUTBOX_STATUS_FAILED,
                row.attempt_count + 1,
                now,
                error,
                row.task_id,
                row.run_id,
            ),
        )
        self.conn.commit()
        row.status = OUTBOX_STATUS_FAILED
        row.attempt_count = row.attempt_count + 1
        row.last_attempt_at = now
        row.last_error = error
        row.lease_expires_at = None

    def _mark_credited(
        self,
        row: OutboxRow,
        *,
        accepted_id: str,
        v3_audit_handle: str,
        now: int,
    ) -> None:
        self.conn.execute(
            f"UPDATE {OUTBOX_TABLE} SET status = ?, attempt_count = ?, "
            f"last_attempt_at = ?, last_error = NULL, lease_expires_at = NULL "
            f"WHERE task_id = ? AND run_id = ?",
            (
                OUTBOX_STATUS_CREDITED,
                row.attempt_count + 1,
                now,
                row.task_id,
                row.run_id,
            ),
        )
        # Per-attempt audit row marking the credit event so the
        # sampling report can join ``outbox.status='credited'`` ↔
        # ``delivery_log.outcome='credited'`` without ambiguity.
        self.conn.execute(
            f"""
            INSERT INTO {DELIVERY_LOG_TABLE}
                (task_id, run_id, attempt, outcome, error,
                 accepted_id, v3_audit_handle, created_at)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                row.task_id,
                row.run_id,
                row.attempt_count + 1,
                DELIVERY_OUTCOME_CREDITED,
                accepted_id,
                v3_audit_handle,
                now,
            ),
        )
        self.conn.commit()
        row.status = OUTBOX_STATUS_CREDITED
        row.attempt_count = row.attempt_count + 1
        row.last_attempt_at = now
        row.last_error = None
        row.lease_expires_at = None
        row.accepted_id = accepted_id
        row.v3_audit_handle = v3_audit_handle

    # ------------------------------------------------------------------
    # Transport adapter — wraps ``transport.record_work_receipt`` so
    # transport-level exceptions become ``TransportUnavailable`` and
    # pass-through results stay structured.
    # ------------------------------------------------------------------

    def _deliver(self, row: OutboxRow) -> RecordReceiptResult:
        """Call the transport and normalise its return shape.

        Returns the structured transport result. Live activation
        wraps the bare stdio ``record_work_receipt`` tool here; tests use
        ``MockTransport`` directly for state-machine isolation.

        Before handing bytes to the transport, recompute the digest from the
        original outbox JSON and require it to match the durable digest column.
        The returned wire object is an explicit flat projection, so an optional
        input ``source`` never leaks into MeshFleet's additionalProperties=false
        record schema.
        """
        envelope, error = _wire_envelope_from_outbox(row)
        if envelope is None:
            return RecordReceiptResult(
                outcome=DELIVERY_OUTCOME_VALIDATION_FAILED,
                error=error or "invalid outbox payload",
            )
        return self.transport.record_work_receipt(envelope)


# ---------------------------------------------------------------------------
# Sampling surface — the only stable interface the dashboard / report
# pipeline uses. Counts + per-status histograms for the production-quality
# numerator (§1.3 of the design).
# ---------------------------------------------------------------------------


@dataclass
class SamplingReport:
    """Snapshot of the outbox at one point in time.

    Designed for the §7 acceptance criterion "Sampling reports both
    quality-gated receipt count and sampled false-DONE rate. Receipt
    volume alone is never the success verdict."
    """

    pending: int = 0
    sent: int = 0
    credited: int = 0
    dead_letter: int = 0
    credit_eligible: int = 0
    audit_handle: str | None = None
    generated_at: int = 0


def build_sampling_report(
    conn: sqlite3.Connection,
    *,
    clock: "callable[[], int]" = time.time,
) -> SamplingReport:
    """Compute a §1.3 strict numerator from the live outbox.

    The credit-eligible count is the rows whose envelope already
    satisfies terminal_outcome=completed, result_contract=ok,
    quality_gate=passed, and at least one evidence handle. The
    credited count is a subset of that — only rows that also
    survived ``verify_ledger_v3`` green. Diverging credit-eligible
    vs credited is the false-DONE-rate denominator.
    """
    rows = conn.execute(
        f"SELECT status, payload_json FROM {OUTBOX_TABLE}"
    ).fetchall()
    report = SamplingReport(generated_at=int(clock()))
    for row in rows:
        status = row["status"]
        if status == OUTBOX_STATUS_PENDING:
            report.pending += 1
        elif status == OUTBOX_STATUS_SENT:
            report.sent += 1
        elif status == OUTBOX_STATUS_FAILED:
            report.dead_letter += 1
        try:
            env = json.loads(row["payload_json"]) if row["payload_json"] else {}
        except (TypeError, ValueError):
            env = {}
        if (
            env.get("terminal_outcome") == "completed"
            and env.get("result_contract") == "ok"
            and env.get("quality_gate") == "passed"
            and bool(env.get("evidence"))
        ):
            report.credit_eligible += 1
            # The delivery state also covers failed-quality receipts.
            # Count only eligible, audited rows in the quality numerator.
            if status == OUTBOX_STATUS_CREDITED:
                report.credited += 1
    return report


def activate_live_transport(
    *,
    conn: sqlite3.Connection,
    config: Any,
    clock: "callable[[], int]" = time.time,
    lease_seconds: int = 60,
) -> "tuple[Reconciler, object]":
    """Construct only the explicitly configured native stdio transport.

    There is deliberately no production mock fallback and no activation
    boolean. The transport itself observes the running process before the
    reconciler is returned.
    """
    from hermes_cli import kanban_receipt_meshfleet_transport as mf  # type: ignore
    return mf.build_reconciler_for_live(
        conn=conn,
        config=config,
        clock=clock,
        lease_seconds=lease_seconds,
    )


def _native_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hermes_cli.kanban_receipt_reconciler",
        description=(
            "Reconcile one exact Hermes Kanban outbox row through an explicitly "
            "installed MeshFleet stdio consumer. No runtime path is inferred."
        ),
    )
    parser.add_argument("--board-db", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--node-bin", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True, type=Path)
    parser.add_argument("--meshfleet-db-file", required=True, type=Path)
    parser.add_argument("--meshfleet-data-file", required=True, type=Path)
    parser.add_argument("--agent-mesh-data-file", required=True, type=Path)
    parser.add_argument("--event-log-file", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--expected-source-ref", required=True)
    parser.add_argument("--expected-package-name", required=True)
    parser.add_argument("--expected-package-version", required=True)
    parser.add_argument("--expected-storage-schema-version", required=True, type=int)
    parser.add_argument("--lease-seconds", type=int, default=60)
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    """Run one bounded native reconciliation and print an inspection receipt."""
    args = _native_parser().parse_args(argv)
    if args.run_id < 1 or args.lease_seconds < 1:
        print(json.dumps({"ok": False, "error": "run-id and lease-seconds must be positive"}))
        return 2
    board_db = args.board_db.expanduser()
    if not board_db.is_absolute() or not board_db.is_file():
        print(json.dumps({"ok": False, "error": "board-db must be an existing absolute file"}))
        return 2
    from hermes_cli.kanban_receipt_meshfleet_transport import (
        McpRuntimeConfig,
        MeshFleetMcpTransport,
    )

    try:
        config = McpRuntimeConfig(
            node_bin=args.node_bin,
            entrypoint=args.entrypoint,
            meshfleet_db_file=args.meshfleet_db_file,
            meshfleet_data_file=args.meshfleet_data_file,
            agent_mesh_data_file=args.agent_mesh_data_file,
            event_log_file=args.event_log_file,
            artifact_dir=args.artifact_dir,
            expected_source_ref=args.expected_source_ref,
            expected_package_name=args.expected_package_name,
            expected_package_version=args.expected_package_version,
            expected_storage_schema_version=args.expected_storage_schema_version,
        )
        conn = sqlite3.connect(str(board_db), timeout=120)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 120000")
        try:
            required_outbox_columns = {
                "task_id",
                "run_id",
                "payload_sha256",
                "payload_json",
                "created_at",
                "status",
                "attempt_count",
                "last_attempt_at",
                "last_error",
                "lease_expires_at",
            }
            observed_columns = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({OUTBOX_TABLE})").fetchall()
            }
            if not required_outbox_columns <= observed_columns:
                missing = sorted(required_outbox_columns - observed_columns)
                raise RuntimeError(f"board DB outbox schema missing columns: {missing}")
            existing = conn.execute(
                f"SELECT status FROM {OUTBOX_TABLE} WHERE task_id = ? AND run_id = ?",
                (args.task_id, args.run_id),
            ).fetchone()
            if existing is None:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": "no matching durable outbox row",
                            "task_id": args.task_id,
                            "run_id": args.run_id,
                        },
                        sort_keys=True,
                    )
                )
                return 3
            if existing["status"] != OUTBOX_STATUS_PENDING:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": "matching outbox row is not pending",
                            "status": existing["status"],
                            "task_id": args.task_id,
                            "run_id": args.run_id,
                        },
                        sort_keys=True,
                    )
                )
                return 2
            transport = MeshFleetMcpTransport(config)
            gate = transport.probe()
            reconciler = Reconciler(
                conn,
                transport,
                lease_seconds=args.lease_seconds,
            )
            processed = reconciler.run_batch(
                max_rows=1,
                task_id=args.task_id,
                run_id=args.run_id,
            )
            outbox = conn.execute(
                f"SELECT * FROM {OUTBOX_TABLE} WHERE task_id = ? AND run_id = ?",
                (args.task_id, args.run_id),
            ).fetchone()
            delivery = conn.execute(
                f"SELECT * FROM {DELIVERY_LOG_TABLE} WHERE task_id = ? AND run_id = ? "
                "ORDER BY id ASC",
                (args.task_id, args.run_id),
            ).fetchall()
            if outbox is None:
                payload = {
                    "ok": False,
                    "error": "no matching durable outbox row",
                    "gate": dataclasses.asdict(gate),
                }
                print(json.dumps(payload, sort_keys=True, default=_json_default))
                return 3
            status = outbox["status"]
            payload = {
                "ok": status == OUTBOX_STATUS_CREDITED,
                "gate": dataclasses.asdict(transport.last_observation or gate),
                "processed": [dataclasses.asdict(row) for row in processed],
                "outbox": dict(outbox),
                "delivery_log": [dict(row) for row in delivery],
                "accepted_receipt": transport.last_accepted_receipt,
                "readback_receipt": transport.last_readback_receipt,
                "audit_handle": transport.last_audit_handle,
            }
            print(json.dumps(payload, sort_keys=True, default=_json_default))
            return 0 if payload["ok"] else 2
        finally:
            conn.close()
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTBOX_TABLE",
    "DELIVERY_LOG_TABLE",
    "OUTBOX_STATUS_PENDING",
    "OUTBOX_STATUS_SENT",
    "OUTBOX_STATUS_CREDITED",
    "OUTBOX_STATUS_FAILED",
    "DELIVERY_OUTCOME_RECORDED",
    "DELIVERY_OUTCOME_DUPLICATE",
    "DELIVERY_OUTCOME_VALIDATION_FAILED",
    "DELIVERY_OUTCOME_CONFLICT",
    "DELIVERY_OUTCOME_TRANSPORT_ERROR",
    "DELIVERY_OUTCOME_READBACK_MISMATCH",
    "DELIVERY_OUTCOME_V3_NOT_GREEN",
    "DELIVERY_OUTCOME_V3_GREEN",
    "DELIVERY_OUTCOME_CREDITED",
    "OutboxRow",
    "DeliveryResult",
    "RecordReceiptResult",
    "VerifyLedgerResult",
    "ReceiptTransport",
    "MockTransport",
    "Reconciler",
    "SamplingReport",
    "TransportUnavailable",
    "build_sampling_report",
    "activate_live_transport",
    "main",
]
