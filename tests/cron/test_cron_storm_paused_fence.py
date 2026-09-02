"""Regression tests for the 2026-09-02 cron-storm (paused-job duplicate fire storm).

Root cause (agents/reports/conductor-cron-storm-20260902T203000Z.md): the
persisted-state stale-error recovery in ``cron/jobs.py`` re-armed a PAUSED
recurring job's ``next_run_at`` to *now* on every tick, ignoring both the
operator's pause markers and the live ``fire_claim`` held by the in-flight
fire that owned the per-job fire fence. Every tick then minted a fresh
'claimed' execution row in ``cron/executions.db``; each duplicate worker
blocked ~30s (``_JOBS_LOCK_TIMEOUT_SECONDS``) on ``_fire_job_lock`` and
failed closed, and the storm repeated while the fence stayed held.

Fixes under test:
  1. ``_job_is_stale_error_recurring`` refuses to re-arm paused/disabled jobs
     (pause markers are authoritative here, exactly as in
     ``_claim_job_for_fire_locked``) — acceptance (1).
  2. The same discriminator refuses to re-arm while a live ``fire_claim``
     exists, so recovery cannot move ``next_run_at`` into immediately-due
     state on every tick while the fire fence is held — acceptance (2).
  3. ``cron/executions.create_execution`` enforces at most one active
     ('claimed'/'running') row per job via a partial unique index and returns
     the existing row on conflict, so duplicate ticks cannot create unbounded
     duplicate claimed rows — acceptance (3).
  4. ``_fire_job_lock`` still exhibits the observed 30s bounded fail-closed
     shape when the fence is held — acceptance (4).

RED on unfixed code: tests 1–3 fail (paused job re-armed / re-armed under a
live claim / unbounded duplicate rows); test 4 pins pre-existing lock
behavior so a future "fix" that removes the fail-closed bound is caught.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron env + a recurring no_agent interval job (10m cadence)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    job = jobs_mod.create_job(
        prompt="probe",
        schedule="every 10m",
        no_agent=True,
        script="probe.py",
    )
    (hermes_home / "scripts" / "probe.py").write_text("print('ok')\n")
    return {"home": hermes_home, "job_id": job["id"]}


def _setup(cron_env, monkeypatch):
    from cron import executions as E
    from cron import scheduler as S
    import cron.jobs as J

    env = cron_env
    monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
    monkeypatch.setattr(S, "_hermes_home", env["home"])
    return S, E, J, env


def _persist_stale_error(J, job_id, *, error_age_minutes=110, **extra):
    """Persist the storm wedge: last_status=error, last_run stale by more than
    a full cadence (10m interval + grace), next_run_at parked in the future."""
    now = datetime.now(timezone.utc)
    fields = {
        "next_run_at": (now + timedelta(minutes=5)).isoformat(),
        "last_status": "error",
        "last_error": "delivery fence held by a wedged fire",
        "last_run_at": (now - timedelta(minutes=error_age_minutes)).isoformat(),
    }
    fields.update(extra)
    J.update_job(job_id, fields)


def _tick(J, S, job_id):
    job = J.get_job(job_id)
    with mock.patch("cron.jobs.load_jobs", return_value=[job]):
        return S.tick(verbose=False, sync=True)


class TestPausedJobNeverRearmedByPersistedErrorRecovery:
    def test_paused_stale_error_job_next_run_untouched_no_execution(
        self, cron_env, monkeypatch
    ):
        """Acceptance (1): a paused recurring job in stale-error state must NOT
        be re-armed by persisted-error recovery — no next_run_at mutation, no
        dispatch, no execution row, on every tick."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        _persist_stale_error(J, job_id, error_age_minutes=110)
        assert J.pause_job(job_id), "fixture must be pausable"
        paused_next = J.get_job(job_id)["next_run_at"]

        for tick_no in range(3):
            _tick(J, S, job_id)
            job = J.get_job(job_id)
            assert job["next_run_at"] == paused_next, (
                f"tick {tick_no}: recovery must not re-arm a paused job "
                f"(next_run_at moved {paused_next} -> {job['next_run_at']})"
            )
            assert J.effective_job_state(job) == "paused"
            # The pause markers are the operator's freeze; the scan must never
            # clear or overwrite them, whatever path it took.
            assert job.get("paused_at"), "tick must not clear paused_at"
            assert job.get("enabled") is False, (
                "tick must not flip a paused job back to enabled"
            )

        assert E.latest_execution(job_id) is None, (
            "a paused job must never dispatch, even in stale-error state"
        )
        from cron import jobs as Jmod
        stats = Jmod.get_persisted_error_recovery_stats()
        assert stats["persisted_error_recoveries"] == 0

    def test_disabled_stale_error_job_next_run_untouched(self, cron_env, monkeypatch):
        """Acceptance (1), disabled variant: enabled=false with no pause
        markers is also frozen and must not be re-armed."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        _persist_stale_error(J, job_id, error_age_minutes=110)
        J.update_job(job_id, {"enabled": False})
        frozen_next = J.get_job(job_id)["next_run_at"]

        _tick(J, S, job_id)
        job = J.get_job(job_id)
        assert job["next_run_at"] == frozen_next
        assert job.get("enabled") is False
        assert E.latest_execution(job_id) is None

    def test_contradictory_half_paused_record_self_disables_no_rearm(
        self, cron_env, monkeypatch
    ):
        """A half-paused record (enabled=true + pause markers — the shape a
        racing pause write can leave) must self-disable and must NOT be
        re-armed or dispatched, on every tick."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        now = datetime.now(timezone.utc)
        _persist_stale_error(
            J,
            job_id,
            error_age_minutes=110,
            state="paused",
            paused_at=now.isoformat(),
            paused_reason="operator freeze mid-storm",
            # leave enabled=True: the contradictory half-paused shape
        )
        parked_next = J.get_job(job_id)["next_run_at"]

        for _ in range(2):
            _tick(J, S, job_id)
            job = J.get_job(job_id)
            assert job["next_run_at"] == parked_next
            assert job.get("enabled") is False, (
                "pause-marker contradiction must self-disable, not fire"
            )
            assert job.get("paused_at")

        assert E.latest_execution(job_id) is None


class TestRecoveryBlockedWhileFireFenceHeld:
    def test_live_fire_claim_blocks_repeated_rearm(self, cron_env, monkeypatch):
        """Acceptance (2): while a fire holds the fence (a live fire_claim on
        the record), recovery must not move next_run_at into immediately-due
        state on every tick. On unfixed code each tick stamps now (or a sooner
        time), the job looks due, and the storm feeds itself."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        now = datetime.now(timezone.utc)
        _persist_stale_error(
            J,
            job_id,
            error_age_minutes=110,
            fire_claim={"at": now.isoformat(), "by": "testhost:4242:claim0"},
        )
        parked_next = J.get_job(job_id)["next_run_at"]

        for tick_no in range(3):
            # get_due_jobs alone is enough: the bug is the recovery's re-arm
            # (a save-side effect of the due scan), not the dispatch itself.
            due = J.get_due_jobs()
            job = J.get_job(job_id)
            assert job["next_run_at"] == parked_next, (
                f"tick {tick_no}: recovery re-armed next_run_at "
                f"({parked_next} -> {job['next_run_at']}) while a live "
                f"fire_claim was held — the storm's self-feeding step"
            )
            assert all(j["id"] != job_id for j in due), (
                "a fenced job must not be returned as due"
            )

        from cron import jobs as Jmod
        stats = Jmod.get_persisted_error_recovery_stats()
        assert stats["persisted_error_recoveries"] == 0

    def test_stale_fire_claim_does_not_block_recovery(self, cron_env, monkeypatch):
        """Guard against over-blocking: a fire_claim older than the TTL means
        the claimant died mid-fire — recovery must still heal the job."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        now = datetime.now(timezone.utc)
        stale_claim_at = now - timedelta(seconds=J._fire_claim_ttl_seconds() + 60)
        _persist_stale_error(
            J,
            job_id,
            error_age_minutes=110,
            fire_claim={"at": stale_claim_at.isoformat(), "by": "testhost:1:dead"},
        )

        _tick(J, S, job_id)
        latest = E.latest_execution(job_id)
        assert latest is not None, (
            "a stale fire_claim must not wedge recovery — the job must "
            "re-dispatch once the claim has expired"
        )
        assert latest["status"] == "completed"


class TestDuplicateClaimedExecutionRowsBounded:
    def test_create_execution_dedups_active_claim_rows(self, cron_env, monkeypatch):
        """Acceptance (3): N concurrent dispatches for one fenced job produce
        exactly ONE active claimed row, not N. The partial unique index +
        existing-row reuse in create_execution is the durable bound."""
        _, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        created = [E.create_execution(job_id, source="builtin") for _ in range(5)]
        ids = {row["id"] for row in created}
        assert len(ids) == 1, (
            f"5 create_execution calls must yield one active claim row, got {ids}"
        )

        rows = E.list_executions(job_id=job_id, limit=50)
        assert len(rows) == 1
        assert rows[0]["status"] == "claimed"

        # After the active row goes terminal, a fresh attempt is a NEW row —
        # the dedup must never block legitimate later fires.
        E.finish_execution(created[0]["id"], success=True)
        second = E.create_execution(job_id, source="builtin")
        assert second["id"] != created[0]["id"]
        rows = E.list_executions(job_id=job_id, limit=50)
        assert len(rows) == 2

    def test_concurrent_create_execution_threads_one_row(self, cron_env, monkeypatch):
        """Threaded shape of the storm: concurrent ticks racing
        create_execution for the same job converge on one row."""
        _, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        results = []
        barrier = threading.Barrier(8)

        def _create():
            barrier.wait(timeout=10)
            results.append(E.create_execution(job_id, source="builtin"))

        threads = [threading.Thread(target=_create) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "create_execution thread wedged"

        ids = {row["id"] for row in results}
        assert len(results) == 8
        assert len(ids) == 1, (
            f"8 concurrent ticks must share one claim row, got {len(ids)}"
        )

    def test_mixed_source_dedup_keeps_first_row(self, cron_env, monkeypatch):
        """A builtin tick racing an external-provider callback for the same
        job must share ONE row; the surviving row keeps the first claim's
        source label (sources are labels, not distinct attempts)."""
        _, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        first = E.create_execution(job_id, source="chronos")
        second = E.create_execution(job_id, source="builtin")
        assert second["id"] == first["id"]
        assert second["source"] == "chronos"
        rows = E.list_executions(job_id=job_id, limit=50)
        assert len(rows) == 1


class TestFireJobLockFailClosedShape:
    def test_second_acquirer_times_out_bounded_and_fails_closed(
        self, cron_env, monkeypatch
    ):
        """Acceptance (4): while one thread holds the fire fence, a second
        acquirer blocks up to the configured bound (the observed 30s shape,
        shortened here via monkeypatch) and then yields False — it must not
        wait forever and must not proceed."""
        _, _, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        monkeypatch.setattr(J, "_JOBS_LOCK_TIMEOUT_SECONDS", 1.0)

        entered = threading.Event()
        release = threading.Event()
        holder_outcome = []

        def _hold_fence():
            with J._fire_job_lock(job_id) as acquired:
                holder_outcome.append(acquired)
                entered.set()
                release.wait(timeout=10)

        holder = threading.Thread(target=_hold_fence)
        holder.start()
        assert entered.wait(timeout=10), "fence holder never entered"

        started = time.monotonic()
        with J._fire_job_lock(job_id) as acquired:
            elapsed = time.monotonic() - started
            assert acquired is False, (
                "a second acquirer must fail closed while the fence is held"
            )
        assert elapsed >= 0.9, (
            f"contender returned after {elapsed:.2f}s — the bounded wait "
            "regressed to instant-fail"
        )
        assert elapsed < 10, (
            f"contender blocked {elapsed:.2f}s — the bounded wait regressed "
            "toward unbounded"
        )

        release.set()
        holder.join(timeout=10)
        assert not holder.is_alive()
        assert holder_outcome == [True], "first acquirer must hold the fence"
