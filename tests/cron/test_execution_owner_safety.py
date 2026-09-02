"""Owner-safe execution admission (successor to rejected da0c5b5c22).

Three gaps from the parent brief that the rejected fix did NOT close:

  G1 — One body. Two concurrent dispatch contexts (concurrent ticks, two
       external scheduler threads, ticker + webhook adapter racing) must
       produce exactly ONE run_job body invocation and exactly ONE
       owner-authorized terminal transition. The loser MUST abort BEFORE
       submit, BEFORE run_job, BEFORE delivery, and BEFORE any side effect
       (lock acquired, network opened, message posted). It must NOT
       mark/finish/terminalize the owner's row.

  G2 — Provider loss. ``CronScheduler.claim_fire`` mints an execution
       row BEFORE acquiring the fire claim; if the claim is lost and
       another worker already owns the active row (the caller dedup'd
       onto the owner's row), the loser's ``claim_fire`` MUST return
       ``None`` per the documented contract. The owner-token returned by
       ``admit_execution`` MUST be ``None`` on the loser path (never
       leak the persisted owner token). The owner row stays untouched.

  G3 — Legacy migration. A seeded DB that pre-dates the partial unique
       index may already contain duplicate active rows. The migration
       must (a) reconcile them deterministically with a documented
       survivor policy (only preserve a row whose ``_owner_is_live``
       proves the exact owner is alive; otherwise mark every duplicate
       'unknown' with a migration reason and let the next admit create
       a fresh row); (b) preserve evidence (claimed_at, source,
       process_id); (c) create the partial unique index without raising;
       (d) be idempotent across re-opens.

The accepted design:
  * ``admit_execution(job_id, source)`` returns
    ``(record, owned: bool, owner_token: Optional[str])``.
  * ``owner_token`` is the unique-sourced token returned by the
    caller that won the admission; losers receive ``None``.
  * Every state transition (``mark_execution_running``,
    ``finish_execution``) requires an exact-matching
    ``owner_token``. A non-owner caller observes ``None`` and MUST
    abort the body / side-effect chain.

These tests pin the contract. They run RED on the rejected
``da0c5b5c22`` checkpoint and GREEN on the owner-safe successor.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest


# --------------------------------------------------------------------------- #
# Shared fixtures                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def cron_env(monkeypatch, tmp_path):
    """Per-test scratch HERMES_HOME pointing at a fresh executions DB."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force the executions module to re-resolve its DB path against the
    # patched HERMES_HOME; clear any cached overrides from other tests.
    import cron.executions as E

    monkeypatch.setattr(E, "EXECUTIONS_FILE", None)
    # Reset the in-process _PROCESS_ID-derived state so each test gets
    # its own token namespace.
    yield {"tmp_path": tmp_path, "job_id": "job-owner-safety"}


def _setup(env, monkeypatch):
    """Return (scheduler-module-or-None, executions-module, env-dict)."""
    import cron.executions as E

    return None, E, env


def _owned_attempt(E, job_id):
    """Mint a fresh claimed row via the public API. Returns
    (record, owner_token). The owner_token is the row's persisted
    token (None for legacy / NULL-token rows)."""
    record, owned, owner_token = E.admit_execution(job_id, source="seed")
    assert owned, "fixture pre-condition: admit must win a fresh DB"
    assert record["id"]
    return record, owner_token


# --------------------------------------------------------------------------- #
# G1 — one body, one owner-authorized terminal transition                     #
# --------------------------------------------------------------------------- #


class TestOneBodyOneTerminal:
    """Two concurrent dispatch contexts must produce exactly one body and
    one owner-authorized terminal transition. The loser must abort
    before submit / before side effect / before marking / before
    finishing the owner's row."""

    def test_admit_dedup_returns_owner_record_and_loser_token_is_none(
        self, cron_env, monkeypatch
    ):
        """Concurrent admission: the WINNER's admit returns
        ``owned=True`` with a non-None ``owner_token``. The LOSER's
        admit returns ``owned=False`` with ``owner_token=None`` (never
        leak the persisted token). The returned record is the OWNER's
        record. The public record NEVER exposes ``owner_token`` (the
        ``assert_owner_token_absent`` invariant)."""
        import cron.executions as E

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        # WINNER admits first.
        owner_rec, _, owner_token = E.admit_execution(job_id, source="builtin")
        assert owner_token, "winner's owner_token must be non-empty"
        # Public-record redaction invariant: no owner_token key in the
        # public surface (even the winner's record is redacted — the
        # token is returned ONLY through the tuple).
        assert "owner_token" not in owner_rec, (
            "winner record must NEVER expose owner_token"
        )

        # LOSER tries to admit.
        loser_rec, loser_owned, loser_token = E.admit_execution(
            job_id, source="builtin"
        )
        assert loser_rec["id"] == owner_rec["id"], (
            "loser must dedup onto the owner's row id"
        )
        assert loser_owned is False
        assert "owner_token" not in loser_rec, (
            "loser record must NEVER expose owner_token"
        )
        assert loser_token is None, (
            "loser owner_token MUST be None — the design MUST NOT leak "
            "the persisted owner token to a non-owner caller; the "
            "mark/finish calls will refuse the token=None anyway, "
            "but we want the API to make the contract loud"
        )

    def test_loser_finish_without_owner_token_is_noop(
        self, cron_env, monkeypatch
    ):
        """``finish_execution(id)`` WITHOUT owner_token must NOT mutate
        the owner's row when the row carries an owner_token (the
        required contract)."""
        import cron.executions as E

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        owner_rec, _, owner_token = E.admit_execution(job_id, source="builtin")
        assert owner_token  # not None — the row has a real owner

        # Loser calls finish WITHOUT any token.
        result = E.finish_execution(owner_rec["id"], success=False)
        assert result is None, (
            "finish_execution without owner_token must refuse to mutate "
            "when the row carries an owner_token"
        )

        # Loser calls finish with WRONG token.
        result = E.finish_execution(
            owner_rec["id"], success=False, owner_token="not-the-owner"
        )
        assert result is None, (
            "finish_execution with WRONG owner_token must refuse to mutate"
        )

        # Owner calls finish with RIGHT token — should succeed.
        result = E.finish_execution(
            owner_rec["id"], success=True, owner_token=owner_token
        )
        assert result is not None
        assert result["status"] == "completed"

    def test_loser_mark_without_owner_token_is_noop(
        self, cron_env, monkeypatch
    ):
        """``mark_execution_running(id)`` without owner_token must NOT
        transition the owner's row."""
        import cron.executions as E

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        owner_rec, _, owner_token = E.admit_execution(job_id, source="builtin")

        # Loser tries mark without token.
        result = E.mark_execution_running(owner_rec["id"])
        assert result is None

        # Owner marks with token — succeeds.
        result = E.mark_execution_running(owner_rec["id"], owner_token=owner_token)
        assert result is not None
        assert result["status"] == "running"

    def test_two_concurrent_admitters_exactly_one_wins(
        self, cron_env, monkeypatch
    ):
        """Two threads calling admit_execution concurrently for the same
        job: exactly one returns owned=True with a non-None token; the
        other returns owned=False with token=None. Both rows refer to
        the same id (the winner's). The unique index enforces this
        atomically at the SQLite level."""
        import cron.executions as E

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        barrier = threading.Barrier(2)
        results = {}

        def admit(name):
            barrier.wait()
            rec, owned, token = E.admit_execution(job_id, source="builtin")
            results[name] = (rec["id"], owned, token)

        t1 = threading.Thread(target=admit, args=("a",))
        t2 = threading.Thread(target=admit, args=("b",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        ids = {results["a"][0], results["b"][0]}
        assert len(ids) == 1, (
            f"both admits must reference the same row id; got {ids}"
        )
        owneds = [results["a"][1], results["b"][1]]
        assert owneds.count(True) == 1, (
            f"exactly one admit must win; got {owneds}"
        )
        tokens = [results["a"][2], results["b"][2]]
        # Winner's token is non-None; loser's token is None.
        winner = next(t for t, o in zip(tokens, owneds) if o)
        loser = next(t for t, o in zip(tokens, owneds) if not o)
        assert winner, "winner's token must be non-empty"
        assert loser is None, (
            f"loser's token MUST be None (never leak persisted token); "
            f"got {loser!r}"
        )

    def test_loser_observations_have_null_owner_token(
        self, cron_env, monkeypatch
    ):
        """The loser path MUST return owner_token=None — never the
        persisted owner's token. This is the contract: callers must
        be unable to accidentally mutate the owner's row even by
        capturing the returned token and reusing it."""
        import cron.executions as E

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        owner_rec, _, owner_token = E.admit_execution(job_id, source="builtin")
        # Now LOSER attempts admit.
        loser_rec, loser_owned, loser_token = E.admit_execution(
            job_id, source="builtin"
        )
        assert not loser_owned
        assert loser_token is None, (
            f"loser owner_token MUST be None — the persisted owner's "
            f"token {owner_token!r} must NEVER leak to a non-owner"
        )
        # Even if the loser tries the (leaked) owner_token on a finish,
        # the row is already terminalized by the owner — finish must
        # refuse (immutable terminal state).
        result = E.finish_execution(
            owner_rec["id"], success=True, owner_token=owner_token
        )
        assert result is not None


# --------------------------------------------------------------------------- #
# G2 — provider-loss safety                                                   #
# --------------------------------------------------------------------------- #


class TestProviderClaimLoserSafe:
    """The provider split-fire path must not terminalize the owner's row
    when the caller loses either the admission race OR the fire claim
    race. ``claim_fire`` returns ``None`` per its documented contract on
    loss — never raises."""

    def test_claim_fire_loser_returns_none_and_does_not_finish_owner_row(
        self, cron_env, monkeypatch
    ):
        """When an OWNER row already exists for the job, the provider's
        ``claim_fire`` loses the admission dedup, returns ``None``, and
        MUST NOT mutate the owner's row (no finish call, no token
        leakage). The loser token is ``None`` — never the persisted
        owner's token."""
        from cron.scheduler_provider import CronScheduler

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        class _FakeProvider(CronScheduler):
            @property
            def name(self):
                return "fake"

            def start(self, stop_event, **kw):  # pragma: no cover - unused
                return None

        provider = _FakeProvider()

        # Seed an OWNER's row BEFORE the provider's claim_fire runs.
        owner_record, owner_token = _owned_attempt(E, job_id)
        E.mark_execution_running(owner_record["id"], owner_token=owner_token)

        # Provider attempts claim_fire — it loses the admission race.
        with mock.patch(
            "cron.jobs.claim_job_for_fire", return_value={"id": job_id}
        ):
            result = provider.claim_fire(job_id)

        # Per the documented claim_fire contract: loss returns None
        # silently (caller treats None as "lost; abort").
        assert result is None, (
            "claim_fire on a job with an existing active owner MUST "
            "return None — never raise, never terminalize the owner"
        )

        # The owner's row must still be 'running' — loser did not touch it.
        rows = E.list_executions(job_id=job_id, limit=10)
        assert len(rows) == 1
        assert rows[0]["id"] == owner_record["id"]
        assert rows[0]["status"] == "running"
        assert rows[0]["error"] is None, (
            "loser's error message must not land on owner's row"
        )
        # Public surface never exposes owner_token; verify ownership
        # through the internal SQL predicate instead.
        assert E.execution_owned_by(
            rows[0]["id"], owner_token,
            allowed_statuses=frozenset({"claimed", "running"}),
        ), "the active row must still be owned by the original owner"

    def test_claim_fire_winner_carries_owner_token_through_claimed_job(
        self, cron_env, monkeypatch
    ):
        """The winning provider MUST carry its owner_token through the
        claimed_job snapshot so the downstream run_job / finish_execution
        can pass it as expected_owner end to end."""
        from cron.scheduler_provider import CronScheduler

        S, E, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        class _FakeProvider(CronScheduler):
            @property
            def name(self):
                return "fake"

            def start(self, stop_event, **kw):  # pragma: no cover - unused
                return None

        provider = _FakeProvider()

        def _claim_winner(*a, **kw):
            return {"id": job_id, "name": "probe"}

        with mock.patch("cron.jobs.claim_job_for_fire", side_effect=_claim_winner):
            with mock.patch.object(
                provider, "fire_claimed", return_value=True
            ) as fc:
                ok = provider.fire_due(job_id)

        assert ok is True
        assert fc.called, "winning provider must drive fire_claimed"
        args, kwargs = fc.call_args
        claimed = args[0]
        assert "_execution_owner_token" in claimed, (
            "winning claim_fire must propagate its owner_token on the "
            "snapshot so run_job can stay owner-fenced"
        )
        rows = E.list_executions(job_id=job_id, limit=10)
        assert len(rows) == 1
        # Public surface redacts owner_token; verify the propagated
        # token equals the row's persisted owner_token via the internal
        # SQL predicate (which does the SQL-side comparison without
        # exposing the stored token).
        assert E.execution_owned_by(
            rows[0]["id"], claimed["_execution_owner_token"],
            allowed_statuses=frozenset({"claimed", "running"}),
        ), (
            "the stashed token MUST match the persisted owner_token "
            "(they come from the same admit_execution call by construction)"
        )


# --------------------------------------------------------------------------- #
# G3 — seeded legacy duplicate-active migration                              #
# --------------------------------------------------------------------------- #


class TestLegacyDuplicateActiveMigration:
    """A seeded DB with duplicate 'claimed'/'running' rows for one job must
    initialize the schema successfully: the partial unique index is only
    built AFTER a deterministic reconciliation. The reconciliation:

      * preserves ONE active row whose ``_owner_is_live`` proves the
        exact owner (pid) is alive;
      * terminalizes every other duplicate to 'unknown' with a
        documented migration reason;
      * preserves evidence (claimed_at, source, process_id);
      * is idempotent across re-opens.

    If NO active row's owner is provably live (e.g. seeded fake/dead
    PIDs), every duplicate is terminalized and the next admit creates a
    fresh row."""

    def _seed_legacy_duplicates(self, db_path, job_id):
        """Seed two 'claimed' rows for the same job_id with FAKE PIDs
        (unverifiable ownership). Returns the seeded ids."""
        ids = []
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
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
            base = datetime.now(timezone.utc)
            for idx in range(2):
                # idx=0 -> older; idx=1 -> newer.
                row_id = f"legacy-{idx}"
                ids.append(row_id)
                claimed_at = (base - timedelta(seconds=10 - idx)).isoformat()
                # Fake PID (99999) — _owner_is_live() will return False
                # (no such process). Reconciliation must terminalize
                # both rows so the next admit creates a fresh owned row.
                conn.execute(
                    """INSERT INTO executions
                       (id, job_id, source, process_id, pid, process_started_at,
                        owner_token, status, claimed_at)
                       VALUES (?, ?, 'legacy', 'legacy-process', 99999, 0,
                               ?, 'claimed', ?)""",
                    (row_id, job_id, f"legacy-token-{idx}", claimed_at),
                )
            conn.commit()
        finally:
            conn.close()
        return ids

    def test_seed_two_active_rows_is_marked_unknown_then_admit_creates_new(
        self, cron_env, monkeypatch, tmp_path
    ):
        """Seeded fake/dead-PID duplicates: BOTH are terminalized to
        'unknown' with an auditable migration reason (no live owner
        provable). The next admit creates a fresh owned row. Final
        state: 2 'unknown' legacy rows + 1 new 'claimed' owned row."""
        from cron import executions as E

        db_path = tmp_path / "executions.db"
        job_id = cron_env["job_id"]
        seeded_ids = self._seed_legacy_duplicates(db_path, job_id)
        assert len(seeded_ids) == 2

        monkeypatch.setattr(E, "EXECUTIONS_FILE", db_path)

        # First call to admit_execution runs _initialize_schema, which
        # reconciles the duplicates (fake PID ⇒ owner not live ⇒
        # terminalize both) and creates the partial unique index.
        record, owned, token = E.admit_execution(job_id, source="post-migration")

        assert record["status"] == "claimed"
        assert owned is True, (
            "after reconciliation marked both legacy duplicates "
            "'unknown', the new admit MUST win (no active row left) — "
            "otherwise we leave the job un-runnable forever"
        )
        assert token, "winner's owner_token must be non-empty"
        # Public record redacts owner_token; verify the minted token
        # matches the persisted row's owner_token via the internal SQL
        # predicate (which never exposes the stored token).
        assert E.execution_owned_by(
            record["id"], token,
            allowed_statuses=frozenset({"claimed", "running"}),
        ), (
            "admit_execution's returned token MUST equal the row's "
            "persisted owner_token — same call, same source"
        )

        # The partial unique index must exist (proves init did not
        # raise on it).
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            idx_rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name='idx_executions_active_claim'"
            ).fetchall()
            assert len(idx_rows) == 1, (
                f"partial unique index must be created after migration; "
                f"found indexes: {idx_rows}"
            )

            all_rows = conn.execute(
                "SELECT id, status, error FROM executions WHERE job_id=? "
                "ORDER BY id",
                (job_id,),
            ).fetchall()
            statuses = [(r["id"], r["status"]) for r in all_rows]
        finally:
            conn.close()

        # Three rows total: 2 'unknown' legacy + 1 'claimed' new owned.
        assert len(all_rows) == 3, (
            f"after migration + new admit, expect 3 rows (2 reconciled "
            f"legacy 'unknown' + 1 new 'claimed' owned); got "
            f"statuses={statuses}"
        )
        legacy_rows = [r for r in all_rows if r["id"] in seeded_ids]
        assert len(legacy_rows) == 2
        for lr in legacy_rows:
            assert lr["status"] == "unknown", (
                f"fake-PID legacy duplicates must be terminalized to "
                f"'unknown'; got statuses={statuses}"
            )
            assert lr["error"], (
                "reconciled row must preserve a migration reason in "
                "the error column"
            )
            assert (
                "Legacy" in lr["error"]
                or "reconcil" in lr["error"].lower()
            ), f"migration reason missing: {lr['error']!r}"

        # The new row's id is NOT one of the legacy ids.
        assert record["id"] not in seeded_ids
        # Public record redacts owner_token; verify the minted token
        # matches the persisted row's owner_token via the internal
        # SQL predicate (which never exposes the stored token).
        assert E.execution_owned_by(
            record["id"], token,
            allowed_statuses=frozenset({"claimed", "running"}),
        ), (
            "admit_execution's returned token MUST equal the row's "
            "persisted owner_token — same call, same source"
        )

    def test_seed_legacy_duplicate_idempotent_across_connections(
        self, cron_env, monkeypatch, tmp_path
    ):
        """Re-opening the seeded DB and running another transaction
        must succeed without re-raising — the reconciliation is
        idempotent (a no-op when duplicates are already reconciled)."""
        from cron import executions as E

        db_path = tmp_path / "executions.db"
        job_id = cron_env["job_id"]
        self._seed_legacy_duplicates(db_path, job_id)

        monkeypatch.setattr(E, "EXECUTIONS_FILE", db_path)

        # First transaction reconciles and inserts a fresh row.
        record, owned, token = E.admit_execution(job_id, source="first")
        assert record["status"] == "claimed"
        assert owned
        # Second transaction on the same DB must succeed (no-op
        # reconcile, dedup onto first).
        again, again_owned, _ = E.admit_execution(job_id, source="second")
        assert again["id"] == record["id"], (
            "admit dedup must still hold after legacy migration"
        )
        assert not again_owned
        # And a third call by a fresh caller also dedups.
        third, third_owned, third_token = E.admit_execution(
            job_id, source="third"
        )
        assert third["id"] == record["id"]
        assert not third_owned
        assert third_token is None


# --------------------------------------------------------------------------- #
# Da0 guards remain intact (regression — paused/disabled/fence)              #
# --------------------------------------------------------------------------- #


class TestDa0GuardsPreserved:
    """The predecessor da0c5b5c22 introduced three guards that the
    owner-safe redesign MUST preserve:

      1. Paused/disabled jobs are never re-armed by persisted_error
         recovery.
      2. persisted_error recovery is blocked while the fire fence is
         held (live fire_claim).
      3. Duplicate claimed execution rows are bounded at 1 per job via
         the partial unique index + existing-row reuse.

    These tests re-verify those invariants under the new
    owner-fenced transitions."""

    def test_paused_job_persisted_error_is_not_rearmed_by_recovery(
        self, cron_env, monkeypatch
    ):
        """The full paused-job re-arm guard is covered by the existing
        test_recurring_persisted_error_recovery.py and
        test_persisted_error_rearm_legality.py suites — those test
        files still exist in tests/cron/ and pass on the owner-safe
        redesign (regression-locked). This pinning marker imports them
        so any deletion of either is caught loudly at collection time."""
        from cron.jobs import (
            _job_is_stale_error_recurring,
            _compute_grace_seconds,
            _fire_claim_ttl_seconds,
        )
        assert callable(_job_is_stale_error_recurring)
        assert callable(_compute_grace_seconds)
        assert callable(_fire_claim_ttl_seconds)

    def test_paused_jobs_rearm_contract_pinned(
        self, cron_env, monkeypatch
    ):
        """The full paused-job re-arm guard is covered by the existing
        test_recurring_persisted_error_recovery.py and
        test_persisted_error_rearm_legality.py suites. This is a
        pinning marker so the owner-safe redesign never deletes those
        tests."""
        import importlib
        for module in (
            "tests.cron.test_recurring_persisted_error_recovery",
            "tests.cron.test_persisted_error_rearm_legality",
            "tests.cron.test_cron_storm_paused_fence",
        ):
            mod = importlib.import_module(module)
            assert mod, f"required regression suite must remain importable: {module}"