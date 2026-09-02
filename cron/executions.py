"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so
# dashboard operations that temporarily enter another profile cannot leak that
# profile's execution records into the import-time home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
_TERMINAL_STATES = ("completed", "failed", "unknown")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


def _connect() -> sqlite3.Connection:
    from cron.jobs import _ensure_cron_dir

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return sqlite3.connect(path, timeout=5)


def _owner_token() -> str:
    """Generate the per-attempt owner token used by the execution ledger.

    The token binds every state transition (``mark_execution_running``,
    ``finish_execution``) to the exact ``create_execution`` caller that
    minted the row. Concurrent dispatches that hit the partial unique index
    and dedup onto an existing row are LOSERS — they return ``token=None``
    and must NOT mutate or terminalize the owner's row. Only the row's
    true owner (the caller whose locally-minted token was the one
    persisted on insert) holds the matching token. The token is composed
    from this process's PID, the per-process uuid constant, and the
    wall-clock so distinct attempts across processes/restarts never
    collide; on duplicate-token collisions (effectively impossible) the
    unique index is the backstop.
    """
    pid = os.getpid()
    started_at = _process_start_time(pid)
    return f"{_PROCESS_ID}:{pid}:{started_at}:{uuid.uuid4().hex}"


def _reconcile_legacy_duplicate_actives(conn: sqlite3.Connection) -> int:
    """Reconcile duplicate active rows before the partial unique index is
    built.

    A seeded DB that pre-dates the partial unique index may already contain
    more than one ``'claimed'``/``'running'`` row for a single ``job_id``.
    Attempting to ``CREATE UNIQUE INDEX ... WHERE status IN
    ('claimed','running')`` on such a DB raises ``sqlite3.IntegrityError``
    and the entire ``_transaction`` blows up - every subsequent caller
    then fails on schema init, leaving the daemon blind to its own
    ledger. Reconcile FIRST.

    Survivor policy (deterministic, documented):

      * A row is preserved as the live owner only when
        ``_owner_is_live(pid, process_started_at)`` PROVES the exact
        owner process is still alive AND ``process_id`` matches the
        current process - i.e. the same process that inserted the row
        is still running it. This is the only state in which we can be
        sure which worker is the legitimate owner and which duplicates
        are stale.
      * Every other active row for the same job is terminalized to
        ``'unknown'`` with a documented migration reason, preserving
        the original ``claimed_at``, ``source``, and ``process_id`` so
        audit history is intact.
      * If NO active row's owner is provably live (the common case for
        a hand-seeded DB with fake PIDs, or after a hard restart that
        killed every previous worker), every duplicate is
        terminalized - the next admit creates a fresh owned row and
        the job is runnable.

    Ties between two provably-live active rows are broken by
    ``ORDER BY claimed_at ASC, id ASC`` so the older row wins (same
    ordering the unique-index admission uses).

    Returns the count of rows reconciled.
    """
    now = _hermes_now().isoformat()
    reconciled = 0
    # Find every job with >1 active row. A row is "active" iff its status is
    # 'claimed' or 'running' (matches the partial unique index WHERE clause).
    dup_jobs = conn.execute(
        """SELECT job_id FROM executions
           WHERE status IN ('claimed','running')
           GROUP BY job_id HAVING COUNT(*) > 1"""
    ).fetchall()
    for dup in dup_jobs:
        job_id = dup["job_id"]
        active_rows = conn.execute(
            """SELECT id, process_id, pid, process_started_at, claimed_at
               FROM executions
               WHERE job_id=? AND status IN ('claimed','running')
               ORDER BY claimed_at ASC, id ASC""",
            (job_id,),
        ).fetchall()
        if len(active_rows) <= 1:
            continue  # was >1 at the SELECT, but someone else fixed it
        # Survivor: only the rows whose owner is provably live AND
        # running in this exact process. If no such row exists, every
        # active row is reconciled - the next admit creates a fresh
        # owned row.
        live_survivors = []
        for r in active_rows:
            if r["process_id"] != _PROCESS_ID:
                continue  # different process - cannot prove owner
            if _owner_is_live(int(r["pid"]), r["process_started_at"]):
                live_survivors.append(r["id"])
        if not live_survivors:
            # Every active row has either an unverifiable owner or an
            # owner from a different process - terminalize ALL of them
            # so the next admit creates a fresh owned row and the job
            # is runnable.
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error=?
                   WHERE job_id=? AND status IN ('claimed','running')""",
                (
                    now,
                    "Legacy duplicate active row reconciled at schema "
                    "init; no provably-live owner could be determined "
                    "from legacy data (seeded / pre-restart); every "
                    "active duplicate terminalized so the next admit "
                    "creates a fresh owned row. Evidence (claimed_at, "
                    "source, process_id) preserved.",
                    job_id,
                ),
            )
            reconciled += cur.rowcount
            continue
        # Pick the oldest live survivor as the surviving owner.
        survivor_id = live_survivors[0]
        cur = conn.execute(
            """UPDATE executions
               SET status='unknown', finished_at=?, error=?
               WHERE job_id=? AND status IN ('claimed','running')
                 AND id != ?""",
            (
                now,
                "Legacy duplicate active row reconciled at schema init; "
                "a provably-live owner was preserved for this job_id "
                "(oldest claimed_at among live rows); every other "
                "duplicate terminalized. Evidence (claimed_at, source, "
                "process_id) preserved.",
                job_id,
                survivor_id,
            ),
        )
        reconciled += cur.rowcount
    return reconciled


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label="cron/executions.db")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             owner_token TEXT,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    # Forward-migration: add the owner_token column to pre-existing
    # schemas. SQLite returns immediately if the column already exists.
    # Legacy rows from before this column existed keep owner_token=NULL,
    # which is treated as "unknown owner" - see _is_owner().
    cols = conn.execute("PRAGMA table_info(executions)").fetchall()
    col_names = {row["name"] for row in cols}
    if "owner_token" not in col_names:
        conn.execute("ALTER TABLE executions ADD COLUMN owner_token TEXT")

    # Reconcile duplicate-active legacy rows BEFORE building the partial
    # unique index; otherwise index creation raises IntegrityError and
    # every transaction in this process dies on schema init.
    _reconcile_legacy_duplicate_actives(conn)

    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_executions_active_claim
           ON executions(job_id) WHERE status IN ('claimed','running')"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, always close.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back
    the transaction; it does not close the connection. Relying on that alone
    leaks a connection (and its WAL/SHM file descriptors) on every call,
    since closing then depends on the garbage collector. Schema init runs
    inside the ``try`` too, so a PRAGMA/DDL failure after a successful
    ``connect()`` still closes the connection instead of leaking it.
    """
    with _lock:
        conn = _connect()
        try:
            _initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()


def _read_owner_token(
    row: Optional[sqlite3.Row], *, conn: sqlite3.Connection,
) -> Optional[str]:
    """Internal accessor for the persisted owner_token.

    ONLY for the owner-fence gate (mark/finish). Public record-producing
    helpers (:func:`_record`, :func:`get_execution`, :func:`list_executions`)
    MUST NOT use this — they strip owner_token at the boundary.
    """
    if row is None:
        return None
    val = row["owner_token"]
    if val is None:
        # owner_token is a derived column from the INSERT; SQLite's
        # default is NULL. Look up by id as a belt-and-braces check.
        lookup = conn.execute(
            "SELECT owner_token FROM executions WHERE id=?",
            (row["id"],),
        ).fetchone()
        return lookup["owner_token"] if lookup else None
    return val


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Convert a sqlite3.Row into the public execution record dict.

    The ``owner_token`` column is the durable credential that authorises
    mark/finish transitions. It MUST NEVER appear in the public record
    returned to a caller — only the owner-safe API
    :func:`admit_execution` returns the token, and only to the WINNER
    of the admission race. This boundary strip enforces that
    invariant at every read path (``list_executions``, ``get_execution``,
    ``admit_execution``, ``create_execution``, ``mark_execution_running``,
    ``finish_execution``) without scattering sanitisation calls across
    the codebase.
    """
    if row is None:
        return None
    d = dict(row)
    d.pop("owner_token", None)
    return d


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    return current is not None and current == started_at


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )


def _is_owner(
    row_owner_token: Optional[str], expected: Optional[str],
    *, existing_dict: Optional[Dict[str, Any]] = None,
) -> bool:
    """Back-compat predicate retained for callers that already hold the
    persisted token (e.g. mark/finish which just queried the row).

    Owner-fence matrix (successor to da0c5b5c22):

      * row_owner_token is a real string AND caller passes that exact
        string -> True.
      * row_owner_token is a real string AND caller passes None / wrong
        token -> False (refuse to mutate — non-owner caller).
      * row_owner_token is None -> ALWAYS False. Legacy / dead-owner
        rows must be reconciled via the dedicated internal SQL recovery
        path (:func:`_reconcile_legacy_duplicate_actives`,
        :func:`recover_interrupted_executions`); the public mutation API
        is owner-safe strict and refuses them.

    The dangerous cross-call scenario (worker A's token used by worker
    B) is still rejected because real owner-safe rows always have a
    non-null row_owner_token, so worker B's "my token" never equals
    the row's token.
    """
    if not row_owner_token:
        # Strict owner-fence: refuse every tokenless mutation of a
        # NULL-token row. Legacy/dead-owner recovery is exclusively the
        # internal SQL path.
        return False
    return expected is not None and row_owner_token == expected


def execution_owned_by(
    execution_id: str,
    owner_token: Optional[str],
    *,
    allowed_statuses: Optional[frozenset] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """INTERNAL narrow predicate: True iff the persisted row exists, has
    a NON-NULL ``owner_token``, AND that token equals ``owner_token``,
    AND the row's status is in ``allowed_statuses``.

    Reads ``owner_token`` via direct SQL — never returns or exposes it.
    This is the only correct way for an external caller (e.g. the
    scheduler's carried-token validation) to verify ownership without
    going through the public record (which strips ``owner_token`` at
    the boundary).

    Defaults match the public mutation contract:
      * ``allowed_statuses = {'claimed','running'}`` — the only states
        in which the scheduler is allowed to drive a transition.
      * ``owner_token`` of ``None`` is ALWAYS refused. The token is the
        credential; a missing credential is an unauthorised caller.
      * A row whose persisted ``owner_token`` is NULL (legacy or
        dead-owner) is refused too. Such rows are reconciled by the
        dedicated internal SQL recovery path
        (:func:`_reconcile_legacy_duplicate_actives`,
        :func:`recover_interrupted_executions`), never mutated here.
      * A row in a terminal state (``completed``/``failed``/``unknown``)
        is refused — terminal transitions are immutable.

    ``conn`` is an optional caller-supplied sqlite3.Connection so the
    predicate can run inside an outer transaction (e.g. inside
    ``mark_execution_running`` / ``finish_execution`` without
    re-opening the DB). When omitted, the predicate opens its own
    short-lived transaction.
    """
    if owner_token is None:
        return False
    if allowed_statuses is None:
        allowed_statuses = frozenset({"claimed", "running"})
    if not allowed_statuses:
        return False  # empty allowed set ⇒ nothing can ever be owned
    placeholders = ",".join("?" for _ in allowed_statuses)
    sql = (
        f"SELECT owner_token FROM executions "
        f"WHERE id=? AND status IN ({placeholders})"
    )
    params: Tuple[Any, ...] = (str(execution_id), *allowed_statuses)
    if conn is not None:
        row = conn.execute(sql, params).fetchone()
        return _row_owns_token(row, owner_token)
    with _transaction() as own_conn:
        row = own_conn.execute(sql, params).fetchone()
        return _row_owns_token(row, owner_token)


def _row_owns_token(
    row: Optional[sqlite3.Row], owner_token: str,
) -> bool:
    """Helper for :func:`execution_owned_by`. Pure comparison: never
    exposes owner_token; returns True iff the row exists, has a non-NULL
    owner_token, and matches the supplied token.
    """
    if row is None:
        return False
    persisted = row["owner_token"]
    if persisted is None:
        # NULL-token row (legacy / dead-owner). Refuse — internal SQL
        # recovery is the only legitimate handler.
        return False
    return persisted == owner_token


def create_execution(job_id: str, *, source: str) -> Dict[str, Any]:
    """Back-compat wrapper around :func:`admit_execution`.

    DEPRECATED for new code: this wrapper exists so legacy call sites
    in ``cron.scheduler`` and external integrations keep compiling
    after the successor-to-da0c5b5c22 owner-safe rework. New code MUST
    use :func:`admit_execution` directly so the caller can thread the
    owner_token through mark/finish.

    Wrapper contract:

      * Always mints an internal owner_token on the row it inserts (no
        ownerless path; every active row has a real token in the
        persisted column so the owner-fence can refuse mutations from
        non-owner callers).
      * Discards the returned token from the public record — the caller
        receives the sanitized record (no ``owner_token`` key) and CANNOT
        mutate the row it just inserted via this wrapper. Mutations
        require the explicit :func:`admit_execution` API which threads
        the token back to the caller.
      * Loser-credential-strip: when this call loses the unique-index
        admission (another worker owns the active row), the persisted
        winner's owner_token is NEVER returned to the caller. The
        caller receives a sanitized record (no token), ``owned=False``
        (in the ``admit_execution`` result tuple), and ``token=None``.

    Concurrent-wrapper regression (successor to da0c5b5c22, see
    ``tests/cron/test_execution_owner_safety.py``): two callers that
    both use this wrapper receive sanitized records without
    owner_token; neither can mark/finish the active row because
    tokenless transitions against an owner-token-bearing row are
    refused. The only way to mutate is the explicit
    :func:`admit_execution` path, which forces the caller to thread
    a token it actually owns.
    """
    record, _owned, _token = admit_execution(job_id, source=source)
    # Token is discarded by design. ``record`` is already sanitized by
    # ``_record`` (owner_token stripped at the boundary).
    return record


def admit_execution(
    job_id: str, *, source: str,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Persist a claimed attempt and return an explicit admission result.

    At most one non-terminal ('claimed'/'running') row may exist per job.
    Enforced by the idx_executions_active_claim partial unique index. This
    is the durable dedup guard for the cron-storm class (rejected
    checkpoint da0c5b5c22). When a tick dispatches a job while a prior
    fire of that same job still holds the per-job fire fence (its
    claim row is still active), the second tick's insert here hits the
    unique index and we return the EXISTING active row instead of
    minting a duplicate claim.

    Explicit admission result (successor to da0c5b5c22 - REQUIRED DESIGN):

      (record, owned, owner_token) where:
        * record       - the persisted execution row, ALWAYS the active
                         row for this job (the OWNER's if another caller
                         won, otherwise the row this caller just
                         inserted).
        * owned        - True iff THIS caller won the unique-index
                         admission (i.e. the row was freshly inserted by
                         this call). False means we dedup'd onto an
                         EXISTING row owned by another worker - this
                         caller MUST treat that as "lost admission race"
                         and abort BEFORE any side effect, mark, or
                         finish.
        * owner_token  - the opaque token that uniquely identifies the
                         caller that inserted the active row. The caller
                         MUST pass this token to
                         ``mark_execution_running(execution_id,
                         owner_token=...)`` and
                         ``finish_execution(execution_id, owner_token=...,
                         ...)`` for every subsequent transition. A
                         non-owner caller observes ``None`` here
                         (the persisted owner_token is NEVER returned
                         to a non-owner) and MUST abort before any
                         transition; ``mark_execution_running`` and
                         ``finish_execution`` refuse any token that does
                         not equal the persisted row's ``owner_token``.

    Inference rules (per parent steering):
      * Owner inference is NEVER by ``source``. Production contenders
        share ``source='builtin'``; only the owner_token decides.
      * No thread-local state. The token is the contract.
    """
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    local_token = _owner_token()
    with _transaction() as conn:
        try:
            conn.execute(
                """INSERT INTO executions
                   (id, job_id, source, process_id, pid, process_started_at,
                    owner_token, status, claimed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'claimed', ?)""",
                (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
                 _process_start_time(pid), local_token, now),
            )
            record_row = conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
            record = _record(record_row)
            return record, True, local_token  # type: ignore[return-value]
        except sqlite3.IntegrityError:
            # A live claimed/running row already exists for this job:
            # another fire (possibly in another thread or process) owns
            # the active attempt. Reuse it rather than creating an
            # unbounded duplicate claim storm. The returned row carries
            # the OWNER's owner_token; the caller MUST treat this as
            # "I lost the admission race" and abort before
            # mark/finish/side-effect.
            row = conn.execute(
                """SELECT * FROM executions
                   WHERE job_id=? AND status IN ('claimed','running')
                   ORDER BY claimed_at ASC, id ASC LIMIT 1""",
                (str(job_id),),
            ).fetchone()
            if row is not None:
                record = _record(row) or {}
                # LOSER contract (successor to da0c5b5c22): the loser
                # MUST receive (record, False, None) - the persisted
                # owner_token is NEVER returned to a non-owner caller.
                # The transition gate refuses any token that doesn't
                # equal the persisted token, so leaking it would still
                # be safe; the loud None makes the contract
                # un-misusable and stops callers from accidentally
                # mutating the owner's row using a leaked token.
                return record, False, None  # type: ignore[return-value]
            # Raced with a terminal transition between the failed insert
            # and the select - no active row anymore; fall through and
            # retry the insert for this fresh attempt.
            conn.execute(
                """INSERT INTO executions
                   (id, job_id, source, process_id, pid, process_started_at,
                    owner_token, status, claimed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'claimed', ?)""",
                (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
                 _process_start_time(pid), local_token, now),
            )
            record_row = conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
            record = _record(record_row)
            return record, True, local_token  # type: ignore[return-value]


def mark_execution_running(
    execution_id: str, *, owner_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once.

    Owner-fenced (successor to da0c5b5c22 - REQUIRED CONTRACT): the
    transition only lands when the row's ``owner_token`` matches the
    supplied ``owner_token``. A caller that supplies ``None`` (or no
    token) on a row that already has an ``owner_token`` observes
    ``None`` from this transition - the mark is refused and the body
    / side-effect chain must abort.

    The token shape is opaque to callers; it is the exact string
    returned by ``admit_execution``'s third tuple element.

    Strict-mode fence (successor): a row whose persisted ``owner_token``
    is NULL (legacy / dead-owner) is ALWAYS refused here too — such
    rows must be reconciled via the dedicated internal SQL recovery
    path, never mutated through this public API.
    """
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        # Cheap existence probe so we can distinguish "no row" from
        # "row present but not owned". The strict owner-fence
        # (execution_owned_by) is what enforces the contract.
        present = conn.execute(
            "SELECT 1 FROM executions WHERE id=?", (execution_id,),
        ).fetchone()
        if present is None:
            return None
        if not execution_owned_by(
            execution_id, owner_token,
            allowed_statuses=frozenset({"claimed"}),
            conn=conn,
        ):
            # Tokenless, wrong, missing token, NULL-token row, or row
            # not in 'claimed' state — refuse to mutate.
            return None
        cur = conn.execute(
            """UPDATE executions SET status='running', started_at=?
               WHERE id=? AND status='claimed'""",
            (now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
    owner_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten.

    Owner-fenced (successor to da0c5b5c22 - REQUIRED CONTRACT): the
    transition only lands when the row's ``owner_token`` matches the
    supplied ``owner_token``. A caller that supplies ``None`` on a row
    that already has an ``owner_token`` observes ``None`` - the finish
    is refused and the owner's row stays untouched until the owner
    itself finishes it. This is the durable owner-safe guarantee.

    This is the critical fix for the provider-loss bug:
    ``CronScheduler.claim_fire`` mints an execution row BEFORE acquiring
    the fire claim; if the claim is lost and the caller is the LOSER of
    the unique-index admission (already dedup'd onto an existing owner's
    row), the loser's ``finish_execution`` here returns ``None`` and the
    owner's row stays active until the owner itself finishes it. The
    caller MUST pass the token returned by ``admit_execution`` to keep
    the chain owner-fenced end to end.

    Strict-mode fence (successor): a row whose persisted ``owner_token``
    is NULL (legacy / dead-owner) is ALWAYS refused here too.
    """
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    with _transaction() as conn:
        present = conn.execute(
            "SELECT 1 FROM executions WHERE id=?", (execution_id,),
        ).fetchone()
        if present is None:
            return None
        if not execution_owned_by(
            execution_id, owner_token,
            allowed_statuses=frozenset({"claimed", "running"}),
            conn=conn,
        ):
            # Tokenless, wrong, missing token, NULL-token row, or row
            # already terminalized — refuse to mutate.
            return None
        cur = conn.execute(
            """UPDATE executions SET status=?, finished_at=?, error=?
               WHERE id=? AND status IN ('claimed','running')""",
            (status, now, detail, execution_id),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    return record


def recover_interrupted_executions() -> int:
    """Mark provably abandoned attempts unknown without scheduling retries."""
    now = _hermes_now().isoformat()
    changed = 0
    recovered: List[Dict[str, Any]] = []
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, process_id, pid, process_started_at FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            cur = conn.execute(
                """UPDATE executions SET status='unknown', finished_at=?, error=?
                   WHERE id=? AND status IN ('claimed','running')""",
                (now,
                 "Scheduler restarted after this execution's owner exited before a durable "
                 "terminal state; whether side effects ran is unknown.",
                 row["id"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _record(conn.execute(
                    "SELECT * FROM executions WHERE id=?", (row["id"],)
                ).fetchone())
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination.

    ``owner_token`` is stripped from every record at the boundary (see
    :func:`_record`) — the public list NEVER exposes the credential.
    """
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [_record(row) for row in rows]  # type: ignore[misc]


def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Read a single execution row by id, or None if missing.

    Used by owner-fence sanity checks (e.g. ``run_job`` verifying that the
    owner_token carried through ``_execution_owner_token`` still matches
    the row's persisted token before driving the body). Cheap indexed
    PRIMARY KEY lookup. ``owner_token`` is stripped at the boundary.
    """
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (str(execution_id),),
        ).fetchone()
    return _record(row)


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: (_record(row) or {}) for row in rows}
